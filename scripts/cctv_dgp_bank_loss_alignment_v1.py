"""Retained loss arithmetic and hypothetical Adam direction; no state mutation."""
COMPONENTS=['MSE','SSIM','ArcFace','detail','anchors','barrier','brightness','clear_MSE','degraded_MSE']


def components(terms, old_terms, arc, old_arc, photo, normalizers, anchors, weights):
    import torch
    raw=[terms['MSE'].mean(),terms['SSIM_loss'].mean(),arc.mean(),terms['landmark_structure'][1:].mean()]
    values=[w*a/s for w,a,s in zip(weights,raw,normalizers)]
    values += [terms['MSE'][0]/anchors[0]+terms['MSE'][1]/anchors[1],
        5*(torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/normalizers[0]
           +torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/normalizers[1]
           +torch.relu(arc-old_arc-1e-6).mean()/normalizers[2]),photo,
        terms['MSE'][0]/anchors[0],terms['MSE'][1:].mean()/normalizers[0]]
    assert len(values)==9 and all(a.ndim==0 and torch.isfinite(a) for a in values)
    return values


def hypothetical_adam_direction(parameters, gradient_rows, stored_optimizer=None):
    """One reference's analytical direction, not an actual accumulated-batch step."""
    import torch
    assert len(gradient_rows)==9
    total=[sum(row[i].detach() for row in gradient_rows[:7]) for i in range(len(parameters))]
    norm=torch.stack([g.double().square().sum() for g in total]).sum().sqrt()
    factor=(1/(norm+1e-6)).clamp(max=1).float()
    if stored_optimizer is not None:
        assert len(stored_optimizer['param_groups'])==1
        group=stored_optimizer['param_groups'][0]
        assert group['betas']==(0.9,0.999) and group['eps']==1e-8 and group['weight_decay']==1e-5
        assert not group.get('amsgrad',False) and not group.get('maximize',False)
        assert len(group['params'])==len(parameters)
    directions=[]
    for i,(parameter,g) in enumerate(zip(parameters,total)):
        # Coupled weight decay follows the recorded PyTorch Adam configuration.
        effective=g*factor+parameter.detach()*1e-5
        if stored_optimizer is None:
            moment=torch.zeros_like(effective); variance=torch.zeros_like(effective); step=1
        else:
            state=stored_optimizer['state'][group['params'][i]]
            assert int(state['step'].item())==50
            moment=state['exp_avg'].to(effective.device); variance=state['exp_avg_sq'].to(effective.device); step=51
        next_m=.9*moment+.1*effective
        next_v=.999*variance+.001*effective.square()
        # Omit the common positive learning rate: this is a direction, not an update.
        direction=-(next_m/(1-.9**step))/((next_v/(1-.999**step)).sqrt()+1e-8)
        assert torch.isfinite(direction).all()
        directions.append(direction)
    return directions, float(norm), float(factor)


def summarize(gradient_rows,directions,names):
    import torch
    groups={'all':list(range(len(names))),
        'decoder':[i for i,n in enumerate(names) if n.startswith('current.')],
        'style':[i for i,n in enumerate(names) if n.startswith('style_conditioner.')],
        'spatial':[i for i,n in enumerate(names) if n.startswith('spatial_conditioners.')]}
    groups.update({'fusion'+scale:[i for i,n in enumerate(names) if n.startswith('fusions.'+scale+'.')]
                  for scale in ['16','32','64','128','256']})
    result={}
    for label,indices in groups.items():
        assert indices
        gram=[[float(sum((gradient_rows[a][i].double()*gradient_rows[b][i].double()).sum() for i in indices))
               for b in range(9)] for a in range(9)]
        direction_dot=[float(sum((gradient_rows[a][i].double()*directions[i].double()).sum() for i in indices)) for a in range(9)]
        result[label]=dict(parameter_names=[names[i] for i in indices],Gram=gram,
            Adam_direction_dot_components=direction_dot,
            direction_squared_norm=float(sum(directions[i].double().square().sum() for i in indices)))
    return result
