"""Bounded closed terminal ZIP admission before exclusive commit; never unpickles."""
from pathlib import Path,PurePosixPath
import time,zipfile

def need(ok,message):
    if not ok:raise ValueError(message)
def admit_terminal_zip(path,*,guard):
    path=Path(path);guard();need(path.is_file() and not path.is_symlink() and path.stat().st_size<=1024**3,'Bounded terminal partial required')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();need(0<len(entries)<=1615 and len({v.filename for v in entries})==len(entries),'Bounded unique terminal ZIP members')
        roots=set();pickle=[];storage=0
        for row in entries:
            guard();name=PurePosixPath(row.filename)
            need(not row.is_dir() and not name.is_absolute() and '..' not in name.parts and ':' not in row.filename and '\\' not in row.filename and len(name.parts)>=2 and str(name)==row.filename,'Safe terminal ZIP namespace')
            need(not row.flag_bits&1 and row.compress_type==zipfile.ZIP_STORED and len(row.filename.encode('utf-8'))+len(row.extra)+len(row.comment)<=512,'Bounded terminal ZIP names/fields/stored members')
            roots.add(name.parts[0]);relative='/'.join(name.parts[1:])
            need(relative in ('data.pkl','byteorder','version','.data/serialization_id','.format_version','.storage_alignment') or relative.startswith('data/') and relative[5:].isdigit(),'Only Torch tensor-state archive members')
            if relative=='data.pkl':pickle.append(row)
            if relative.startswith('data/'):storage+=1
            need(row.file_size<=1024**3,'Bounded terminal ZIP member')
        need(len(roots)==1 and len(pickle)==1 and pickle[0].file_size<=1024**2 and storage<=1609 and len(archive.comment)<=1024,'Terminal pickle/storage metadata caps')
        # Stream every member under the external save deadline; ZipExtFile validates CRC at EOF.
        for row in entries:
            with archive.open(row) as stream:
                copied=0
                for block in iter(lambda:stream.read(1024*1024),b''):
                    copied+=len(block);guard();need(copied<=row.file_size,'ZIP member expanded beyond declared extent')
                need(copied==row.file_size,'Complete terminal ZIP member CRC read')
    guard();return {'complete':True,'ZIP_members':len(entries),'storage_records':storage,'primitive_pickle_bytes':pickle[0].file_size,'all_member_CRC_pass':True,'unpickled':False,'bounded_during_all_writes':True,'internal_caps_checked_after_close_before_commit':True}
