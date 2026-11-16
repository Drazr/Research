"""Explicit immutable recovery references; no tree scans or mutable latest file."""
from src.stage2.schema import plain, frozen, check
from .artifacts import Store, write
from .readouts import _body, candidate, FittedSystem, load_statistics, expected_rows

KINDS={'candidates':'candidate','selections':'selection','statistics':'statistics',
       'assignments':'assignments','control_scores':'control_scores','null_groups':'null_group','gates':'g0_gate'}


def read_completed(plan,ref,kind,expected):
    check(kind in KINDS.values(),'Unsupported recovery artifact')
    store=Store(plan.root,plan.protocol)
    b,_=_body(plan,ref,kind,store,fit=kind!='statistics')
    for k,v in expected.items(): check(b.get(k)==plain(v),'Wrong completed block: '+k)
    store.complete(ref,'stage3_'+kind)
    if kind=='candidate': candidate(plan,ref,store)
    elif kind=='selection': FittedSystem(plan,ref,'source_select',expected)
    elif kind=='statistics':
        check('group_id' in expected,'Statistics require explicit group')
        load_statistics(plan,ref,expected['group_id'],expected_rows(plan,expected['group_id'],'FIT',store),store)
    return frozen(b)


def publish_run_index(plan,entries,producer):
    check(set(entries)<=set(KINDS),'Unknown run-index category')
    store=Store(plan.root,plan.protocol); refs={}
    for name,items in entries.items():
        refs[name]=[]; seen=set()
        for ref in items:
            b,_=_body(plan,ref,KINDS[name],store,fit=name!='statistics')
            r=store.ref(ref)
            check(r['sha256'] not in seen,'Duplicate run-index reference');seen.add(r['sha256'])
            # An index binds already completed units; it is not another whole-
            # payload acceptance scan. Consumers verify the units they reuse.
            refs[name].append(r)
    return write(plan,'results/'+plan['fit_identity'][:16]+'/indexes','run_index',
                 {'entries':refs,'counts':{k:len(v) for k,v in refs.items()},'stage3_complete':False},{},producer)


def load_run_index(plan,ref):
    b,store=_body(plan,ref,'run_index')
    check(set(b['entries'])<=set(KINDS),'Unknown run-index schema')
    for name,items in b['entries'].items():
        check(b['counts'][name]==len(items),'Index count mismatch')
        for item in items: store.marker(item,'stage3_'+KINDS[name])
    return frozen(b['entries'])
