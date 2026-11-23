"""O(1) evidence-budget ledger copy for the prospective predictor CE pilot.

The immutable coverage-v2 implementation and its failed return remain untouched.
This helper changes accounting cost, not declared deadline/storage policy.
Every open validates a canonical path. Each write is checked through the owned
file handle, with cached category totals and logical file extents. No public
fileno is exposed, so native serializers must use the bounded write callback.
"""
from dataclasses import dataclass
from pathlib import Path
import json,shutil,time
from research.completion_codeformer_predictor_ce_v1.contracts import CONTRACT,need

@dataclass
class Entry:
 path:Path
 category:str
 extent:int=0

class OutputBudget:
 def __init__(self,out,started,*,disk_free=None,clock=None):
  original_out=Path(out)
  need(not any(part.is_symlink() for part in [original_out,*original_out.parents]),'Linked output root or parent refused')
  self.out=original_out.resolve(strict=True)
  need(self.out.is_dir(),'Output root must be a real existing directory')
  self.started=started;self.clock=clock or time.monotonic
  self.disk_free=disk_free or (lambda:shutil.disk_usage(self.out).free)
  self.sizes={};self.entries={};self.totals={kind:0 for kind in ('terminal','teacher','metadata','PNG','array')};self.extent_total=0
  self.metadata_external=[];self.terminal_committed=False;self.supervisor_reserve=1024**2
  self.check_counts={'canonical_paths':0,'path_components':0,'guard_calls':0,'write_checks':0,'external_stats':0}
 def canonical(self,path):
  self.check_counts['canonical_paths']+=1
  path=Path(path)
  need('..' not in path.parts,'Path traversal refused')
  resolved=path.resolve(strict=False)
  need(resolved!=self.out and resolved.is_relative_to(self.out),'Output writer escaped evidence root')
  for part in [path,*path.parents]:
   self.check_counts['path_components']+=1
   need(not part.is_symlink(),'Linked output path refused')
   if part.resolve(strict=False)==self.out:break
  need(len(('outputs/'+resolved.relative_to(self.out).as_posix()).encode('utf-8'))<=240,'Output relative path UTF8 cap240')
  return resolved
 def classify(self,path):
  rel=path.relative_to(self.out).as_posix()
  if rel in ('terminal_training_state.pth','terminal_training_state.pth.partial'):return 'terminal'
  if rel.startswith('teacher_labels/') and rel.endswith('.npz'):return 'teacher'
  if rel.endswith('.png'):return 'PNG'
  if rel.endswith(('.json','.log')):return 'metadata'
  need(rel.endswith('.npz'),'Unexpected output evidence type');return 'array'
 def category(self,path):
  entry=self.entries.get(Path(path))
  return entry.category if entry is not None else self.classify(self.canonical(path))
 def total(self,category=None):
  return self.extent_total if category is None else self.totals.get(category,0)
 def external_metadata(self):
  total=0
  for path in self.metadata_external:
   self.check_counts['external_stats']+=1
   path=Path(path)
   if path.is_file():total+=path.stat().st_size
  return total
 def guard(self,*,terminal=False,emergency=False):
  self.check_counts['guard_calls']+=1
  limit=1800 if terminal or emergency or self.terminal_committed else 1680
  need(self.clock()-self.started<limit,'Finite worker deadline; preserve terminal state')
  external=self.external_metadata()
  need(self.extent_total+external+self.supervisor_reserve<=CONTRACT['output_bytes'],'Evidence/export metadata cap')
  need(self.totals['teacher']<=CONTRACT['teacher_bytes'],'Teacher label cap')
  need(self.totals['terminal']<=CONTRACT['terminal_bytes'],'Terminal serialization cap')
  need(self.totals['metadata']+external+self.supervisor_reserve<=CONTRACT['metadata_bytes'],'Metadata/log/supervisor cap')
  reserve=(0 if terminal or emergency else 1024**2 if self.terminal_committed else CONTRACT['terminal_bytes']+1024**2)
  need(self.disk_free()>=CONTRACT['stop_free_bytes']+reserve,'Stop before reserved terminal/free disk is consumed')
  return external
 def before_entry_write(self,entry,end,*,terminal=False,emergency=False):
  self.check_counts['write_checks']+=1
  need(self.entries.get(entry.path) is entry and type(end) is int and end>=0,'Unknown owned output or invalid extent')
  external=self.guard(terminal=terminal,emergency=emergency);category=entry.category;growth=max(0,end-entry.extent)
  need(self.extent_total+external+self.supervisor_reserve+growth<=CONTRACT['output_bytes'],'Whole output cap before write')
  if category=='terminal':need(self.totals['terminal']+growth<=CONTRACT['terminal_bytes'],'Terminal write cap')
  elif category=='teacher':need(self.totals['teacher']+growth<=CONTRACT['teacher_bytes'],'Retained teacher cap')
  elif category=='metadata':
   metadata_cap=CONTRACT['metadata_bytes']-self.supervisor_reserve-(0 if emergency else 512*1024)
   need(self.totals['metadata']+external+growth<=metadata_cap,'Metadata write cap with failure receipt reserve')
  elif category=='PNG':need(end<=CONTRACT['PNG_bytes_each'],'PNG extent cap')
  if category!='terminal' and not emergency and not self.terminal_committed:
   need(self.extent_total+external+self.supervisor_reserve+growth<=CONTRACT['output_bytes']-CONTRACT['terminal_bytes']-1024**2,'Preserve terminal byte reservation')
  if self.terminal_committed and not terminal and not emergency:
   need(self.extent_total+external+self.supervisor_reserve+growth<=CONTRACT['output_bytes']-1024**2,'Preserve final failure receipt headroom')
  need(self.disk_free()-growth>=CONTRACT['stop_free_bytes'],'Preserve2GiBfree while writing')
 def before_write(self,path,end,*,terminal=False,emergency=False):
  key=Path(path);entry=self.entries.get(key)
  if entry is None:
   key=self.canonical(path);entry=self.entries.get(key)
  need(entry is not None,'Only opened evidence may be written')
  return self.before_entry_write(entry,end,terminal=terminal,emergency=emergency)
 def commit_extent(self,entry,end):
  need(self.entries.get(entry.path) is entry,'Owned ledger entry changed')
  growth=max(0,end-entry.extent)
  entry.extent=max(entry.extent,end);self.sizes[entry.path]=entry.extent;self.extent_total+=growth;self.totals[entry.category]+=growth
 def open(self,path,*,terminal=False,emergency=False):return CappedWriter(self,path,terminal=terminal,emergency=emergency)
 def rename(self,before,after):
  before=Path(before);entry=self.entries.get(before)
  if entry is None:before=self.canonical(before);entry=self.entries.get(before)
  need(entry is not None,'Unowned evidence rename')
  after=self.canonical(after);need(after not in self.entries,'Rename target is already owned')
  new_category=self.classify(after);need(new_category==entry.category,'Rename cannot change evidence budget category')
  del self.entries[before];del self.sizes[before];entry.path=after;self.entries[after]=entry;self.sizes[after]=entry.extent
  if after.name=='terminal_training_state.pth':self.terminal_committed=True
 def json(self,path,value,*,emergency=False):
  body=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')
  with self.open(path,emergency=emergency) as stream:stream.write(body)
 def assert_ledger_consistent(self):
  """Explicit audit boundary; not called in every write callback."""
  need(self.extent_total==sum(self.sizes.values()),'Total ledger extent mismatch')
  for category,total in self.totals.items():need(total==sum(e.extent for e in self.entries.values() if e.category==category),'Category ledger mismatch')
  need(set(self.entries)==set(self.sizes) and all(e.extent==self.sizes[k] for k,e in self.entries.items()),'Entry namespace/size mismatch')
  return True

class CappedWriter:
 def __init__(self,budget,path,*,terminal=False,emergency=False):
  self.budget=budget;self.path=budget.canonical(path);self.terminal=terminal;self.emergency=emergency
  category=budget.classify(self.path);need(self.path not in budget.entries,'Evidence already owned; refusing overwrite')
  need(len(budget.entries)+1+4<=1000,'At most1000 return regular files including protocol/log/supervisor/manifest')
  self.path.parent.mkdir(parents=True,exist_ok=True)
  # Revalidate after parent creation, before exclusive open; never follows links.
  need(budget.canonical(self.path)==self.path,'Output path changed during preparation')
  self.stream=self.path.open('xb');self.entry=Entry(self.path,category)
  budget.entries[self.path]=self.entry;budget.sizes[self.path]=0
 def write(self,data):
  end=self.stream.tell()+len(data);self.budget.before_entry_write(self.entry,end,terminal=self.terminal,emergency=self.emergency)
  n=self.stream.write(data);need(n==len(data),'Short evidence write');self.budget.commit_extent(self.entry,end);return n
 def read(self,*args):return self.stream.read(*args)
 def tell(self):return self.stream.tell()
 def seek(self,*args):return self.stream.seek(*args)
 def flush(self):return self.stream.flush()
 def writable(self):return True
 def seekable(self):return True
 @property
 def closed(self):return self.stream.closed
 def close(self):return self.stream.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
