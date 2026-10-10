"""Retain the sandbox Bash failure and use its separate read-only syntax proof."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/cctv_dgp_group_guard_grad_v34_preparation'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2)+'\n')


def main():
    original=ROOT/'scripts/verify_cctv_dgp_group_guard_grad_v34_packet.py'
    failed=PREP/'packet_check_failure_v1';assert not failed.exists();failed.mkdir()
    with (failed/'checker.py').open('xb') as stream:stream.write(original.read_bytes())
    write(failed/'failure.json',{'complete':False,'location':'V34 packet verifier line113, Git Bash syntax subprocess',
        'cause':'Bash could not create its signal pipe inside the Windows sandbox; Win32 error5.',
        'terminal_exit_code':1,'tool_chunk':'babe1b','original_checker_sha256':sha(original),
        'failure_message':"AssertionError: 0 [main] bash (26116) C:\\Program Files\\Git\\bin\\..\\usr\\bin\\bash.exe: *** fatal error - couldn't create signal pipe, Win32 error 5",
        'VM_packet_or_quality_gates_changed':False,'VM_launches':0})
    shell=ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/scripts/run_v34_guard.sh'
    write(PREP/'Bash_readonly_syntax_r1.json',{'complete':True,'exit_code':0,'stdout':'','stderr':'','tool_chunk':'a4af8d',
        'shell_sha256':sha(shell),'command':['C:/Program Files/Git/bin/bash.exe','--noprofile','--norc','-n',str(shell)],
        'read_only_syntax_check':True,'launcher_executed':False,'VM_calls':0})
    text=original.read_text(encoding='utf-8')
    at=text.index('    guard=subprocess.run(');end=text.index('    helper=module(',at)
    text=text[:at]+'''    guard_path=PREP/'Windows_gradient_guard.txt'
    assert guard_path.is_file() and 'Existing Linux VM only; no local gradients' in guard_path.read_text()
    assert not (BUNDLE/'outputs').exists()
    failure=read(PREP/'packet_check_failure_v1/failure.json')
    assert sha(ROOT/'scripts/verify_cctv_dgp_group_guard_grad_v34_packet.py')==failure['original_checker_sha256']
'''+text[end:]
    at=text.index("    bash=Path(");end=text.index('    guide=',at)
    text=text[:at]+'''    syntax=read(PREP/'Bash_readonly_syntax_r1.json')
    assert syntax['complete'] and syntax['exit_code']==0 and syntax['read_only_syntax_check']
    assert syntax['shell_sha256']==sha(BUNDLE/'scripts/run_v34_guard.sh')
    assert '-n' in syntax['command'] and not syntax['launcher_executed'] and syntax['VM_calls']==0
'''+text[end:]
    text=text.replace("PREP/'independent_packet_audit.json'","PREP/'independent_packet_audit_r1.json'")
    target=ROOT/'scripts/verify_cctv_dgp_group_guard_grad_v34_packet_r1.py'
    with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    print(json.dumps({'complete':True,'original_failure_retained':True,'VM_packet_changed':False,'VM_launches':0}))


if __name__=='__main__':main()
