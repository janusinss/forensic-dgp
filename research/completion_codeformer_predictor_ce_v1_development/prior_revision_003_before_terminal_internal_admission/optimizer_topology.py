"""Pure name-based Adam topology projection/merge; no Torch/model imports."""
from copy import deepcopy
def need(ok,message):
    if not ok:raise ValueError(message)
def ordered_ids(state,names):
    need(isinstance(names,list) and len(names)==len(set(names)),'Unique named optimizer order required')
    need(len(state['param_groups'])==1,'Exactly one Adam group required')
    ids=state['param_groups'][0]['params'];need(len(ids)==len(names) and len(ids)==len(set(ids)) and set(ids)==set(state['state']),'Complete optimizer ID/name binding required')
    return dict(zip(names,ids))
def project(parent_state,parent_names,active_names,*,clone):
    mapping=ordered_ids(parent_state,parent_names);need(len(active_names)==len(set(active_names)) and set(active_names)<=set(mapping),'Unique active names must be original code parameters')
    group=deepcopy(parent_state['param_groups'][0]);group['params']=list(range(len(active_names)))
    return {'state':{i:clone(parent_state['state'][mapping[name]]) for i,name in enumerate(active_names)},'param_groups':[group]}
def merge(parent_state,parent_names,active_state,active_names,*,clone):
    original=ordered_ids(parent_state,parent_names);active=ordered_ids(active_state,active_names);need(set(active)<=set(original),'Active optimizer escaped original code topology')
    merged=clone(parent_state)
    for name,index in active.items():merged['state'][original[name]]=clone(active_state['state'][index])
    need(merged['param_groups']==parent_state['param_groups'],'Preserve historical full-code group metadata')
    return merged
