"""Complete inventory validation and independent numeric readback acceptance."""
import json
import subprocess
import sys
from src.stage2.schema import check, plain
from .artifacts import Store, write
from .data import CS
from .readouts import _body, candidate, choose, FittedSystem


def validate_stage3(plan,candidate_registry,selection_registry,g0_receipt,producer):
    check(len(plan['groups'])==145,'Complete frozen source inventory required')
    store=Store(plan.root,plan.protocol); candidates={}; selections={}; summaries={}
    expected={(g,a,c) for g in plan['groups'] for a in ('true','seed0','seed1','seed2') for c in CS}
    for ref in candidate_registry:
        b,*_=candidate(plan,ref,store)
        key=(b['group_id'],b['assignment_id'],b['C'])
        check(key in expected and key not in candidates,'Duplicate/foreign candidate')
        candidates[key]=store.ref(ref)
        summaries[key]={'score':b['metrics']['selection_score'],'statistics':b['statistics']}
    check(set(candidates)==expected,'Missing candidate blocks')
    for ref in selection_registry:
        b,_=_body(plan,ref,'selection',store); key=(b['group_id'],b['assignment_id'])
        check(key not in selections and (key[0],key[1],.01) in candidates,'Duplicate/foreign selection')
        check(b['candidates']==[candidates[(*key,c)] for c in CS], 'Selection candidate coverage mismatch')
        scores={c:summaries[(*key,c)]['score'] for c in CS}
        check(all(summaries[(*key,c)]['statistics']==summaries[(*key,CS[0])]['statistics'] for c in CS),'Candidate scaler mismatch')
        check(b['selected_C']==choose(scores) and b['scores']=={str(c):v for c,v in scores.items()}
              and store.ref(b['selected'])==candidates[(*key,b['selected_C'])],'Invalid C selection')
        # Candidate payloads/statistics were fully checked once above; do not
        # reread every coefficient/DEV-S matrix again merely to check selection.
        selections[key]=store.ref(ref)
    check(set(selections)=={(g,a) for g in plan['groups'] for a in ('true','seed0','seed1','seed2')},'Missing selections')
    gate,_=_body(plan,g0_receipt,'g0_gate',store)
    check(not gate['flagged'] and gate['p']>.01 and gate['structural_checks']=='passed'
          and set(gate['groups'])==set(plan['groups']),'Unresolved/incomplete G0 gate')
    import numpy as np
    maximum=np.zeros(1999); observed=0.
    for gid,ref in gate['groups'].items():
        nb,_=_body(plan,ref,'null_group',store)
        check(nb['group_id']==gid and nb['draw_ids']==list(range(1,2000)), 'Wrong null group')
        aids=set()
        for sr in nb['scores']:
            sb,_=_body(plan,sr,'control_scores',store)
            key=(gid,sb['assignment_id']); check(key in selections and sb['selection']==selections[key], 'G0/selected registry mismatch')
            check(sb['assignment_id'] not in aids,'Duplicate seed score'); aids.add(sb['assignment_id'])
            store.complete(sr,'stage3_control_scores')
        check(aids=={'seed0','seed1','seed2'},'Missing control seed')
        maximum=np.maximum(maximum,store.array(ref,'maxima.npy','<f8',(1999,))); observed=max(observed,nb['observed'])
    check(np.array_equal(maximum,store.array(g0_receipt,'maxima.npy','<f8',(1999,))) and observed==gate['observed']
          and gate['p']==(1+int((maximum>=observed).sum()))/2000,'Global gate reduction mismatch')
    body={'suite':plain(plan['suite']),'candidates':[candidates[k] for k in sorted(candidates)],
          'selections':[selections[k] for k in sorted(selections)],'g0_gate':store.ref(g0_receipt),
          'stage2_groups':plain(plan['groups']),'ordinary_systems':145,'stage3_complete':False}
    pending=write(plan,'results/stage3-validation','registry_readback',body,{},producer)
    cmd=[sys.executable,'-m','src.stage3.operations','readback','--phase1-root',str(plan.root),
         '--manifest',str(plan.root/pending['path'])]
    subprocess.run(cmd,check=True)
    body.update(stage3_complete=True,independent_readback=plain(pending))
    return write(plan,'results/stage3-validation','system_registry',body,{},producer)
