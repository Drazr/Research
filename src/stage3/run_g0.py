"""CPU G0 scheduling over the frozen shared numerical and recovery interfaces."""
import argparse
import json
from pathlib import Path
import time
import threading

from src.common.contracts import ContractError
from src.stage2.schema import check, plain
from .interfaces import (CS, load_fit_plan, open_numeric_session, current_producer,
                         load_or_publish_statistics, fit_candidate_block,
                         select_system, score_g0, evaluate_g0, read_completed,
                         publish_run_index, load_run_index, expected_group_rows)

SEEDS = ('seed0', 'seed1', 'seed2')
KINDS = {'statistics': 'statistics', 'assignments': 'assignments',
         'candidates': 'candidate', 'selections': 'selection',
         'control_scores': 'control_scores', 'null_groups': 'null_group',
         'gates': 'g0_gate'}


def peak_process_bytes():
    """OS process high-water memory, including native numerical allocations."""
    import sys
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ('peak', 'working', 'peak_paged', 'paged', 'peak_nonpaged',
                 'nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
            return counters.peak
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run_g0(plan, groups, seeds, assignments, producer, *, supplied=None,
           resume_indexes=(), statistics_indexes=(), profile=False, emit=None):
    """Process explicit disjoint ownership; missing scalers defer only their group.

    supplied and resume_indexes contain explicit completed references, never paths
    discovered by walking storage. Ordinary indexes contribute statistics only.
    Empty groups performs reduction only. A seed subset never yields a partial
    three-seed null summary. Callers retain streamed indexes on handled failure.
    """
    emit = emit or (lambda value: print(json.dumps(plain(value)), flush=True))
    groups, seeds = list(groups), list(seeds)
    check(type(profile) is bool, 'Profile must be a boolean')
    check(len(groups) == len(set(groups)) and set(groups) <= set(plan['groups']),
          'Explicit unique known group ownership required')
    check(len(seeds) == len(set(seeds)) and set(seeds) <= set(SEEDS),
          'Only fixed seeds 0,1,2 are permitted')
    check(not groups or seeds, 'Fitting ownership requires seeds')
    check(not profile or (set(groups) <= set(plan['profiling_groups']) and seeds == ['seed0']),
          'Profile only metadata representatives at seed0 C=.01')
    started = time.perf_counter()
    entries = {name: [] for name in KINDS}
    identities = {name: {} for name in KINDS}
    bodies = {}
    timings = []
    indexes = []

    def timed(event, operation, **fields):
        wall, cpu = time.perf_counter(), time.process_time()
        emit(dict(event=event + '_start', **fields))
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(30):
                emit(dict(event='progress', phase=event,
                          wall_seconds=time.perf_counter()-wall, **fields))
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            result = operation()
        finally:
            stop.set()
            thread.join()
        row = dict(event=event + '_complete', wall_seconds=time.perf_counter()-wall,
                   cpu_seconds=time.process_time()-cpu, peak_process_bytes=peak_process_bytes(), **fields)
        timings.append(row)
        emit(row)
        return result

    def add(name, ref):
        ref = plain(ref)
        if ref in entries[name]:
            return bodies[(name, ref['sha256'])]
        # Statistics have an input identity independent of control/true identity.
        # Read their group from the shared authenticated body, then invoke its
        # semantic loader with an explicit expected group before any fit can run.
        if name == 'statistics':
            from .artifacts import Store
            body = Store(plan.root, plan.protocol).json(ref, 'body.json', 'stage3_statistics')
            gid = body['group_id']
            check(gid in plan['groups'], 'Foreign source statistics')
            body = read_completed(plan, ref, 'statistics', {'group_id': gid})
            key = gid
        else:
            body = read_completed(plan, ref, KINDS[name], {})
            gid = body.get('group_id')
            if gid is not None:
                check(gid in plan['groups'], 'Foreign G0 group')
            if name in ('candidates', 'selections', 'control_scores'):
                check(body['assignment_id'] in SEEDS, 'G0 requires control targets')
                key = (gid, body['assignment_id'])
                if name == 'candidates':
                    check(body['C'] in CS, 'Unknown control C')
                    key += (body['C'],)
            else:
                key = gid if name == 'null_groups' else name
                if name == 'null_groups':
                    check(len(body['scores']) == 3, 'Null summary requires three seed score references')
        old = identities[name].get(key)
        check(old is None or old == ref, 'Conflicting '+name+' coverage')
        identities[name][key] = ref
        entries[name].append(ref)
        bodies[(name, ref['sha256'])] = body
        return body

    def ingest(mapping):
        check(set(mapping) <= set(KINDS), 'Unknown G0 recovery category')
        for name, refs in mapping.items():
            for ref in refs:
                add(name, ref)

    ingest(supplied or {})
    for ref in resume_indexes:
        ingest(load_run_index(plan, ref))
    for ref in statistics_indexes:
        ingest({'statistics': load_run_index(plan, ref).get('statistics', ())})
    add('assignments', assignments)
    # The shared evaluator authenticates every deterministic vector/mask map;
    # this empty-score call does no fitting, scoring, null computation or writes.
    evaluate_g0(plan, [], entries['assignments'], [], producer)

    def checkpoint(gid, event):
        local = {'assignments': entries['assignments']}
        for name in ('statistics', 'candidates', 'selections', 'control_scores', 'null_groups'):
            local[name] = [r for r in entries[name]
                           if bodies[(name, r['sha256'])].get('group_id') == gid]
        ref = timed('index_write_readback', lambda: publish_run_index(plan, local, producer), group_id=gid)
        indexes.append(plain(ref))
        emit({'event': event, 'group_id': gid, 'index': plain(ref),
              'counts': {k: len(v) for k, v in local.items()}})

    # Recover dependency closure from score/selection refs, so a scores-only
    # handoff cannot cause completed selection/fitting/scoring to be repeated.
    for ref in list(entries['gates']):
        b = bodies[('gates', ref['sha256'])]
        check(set(b['groups']) == set(plan['groups']), 'Incomplete global gate')
        for ref in b['groups'].values():
            add('null_groups', ref)
    for ref in list(entries['null_groups']):
        b = bodies[('null_groups', ref['sha256'])]
        seen_seeds = set()
        for ref in b['scores']:
            sb = add('control_scores', ref)
            check(sb['group_id'] == b['group_id'], 'Null score group mismatch')
            check(sb['assignment_id'] not in seen_seeds, 'Duplicate null score seed')
            seen_seeds.add(sb['assignment_id'])
        check(seen_seeds == set(SEEDS), 'Incomplete null seed coverage')
    for ref in list(entries['control_scores']):
        b = bodies[('control_scores', ref['sha256'])]
        sb = add('selections', b['selection'])
        check(all(b[k] == sb[k] for k in ('group_id', 'assignment_id')), 'Score selection mismatch')
        # Saved logits are already checksum-verified. Recover exact row identity
        # from authenticated metadata instead of reloading unused SCREEN vectors.
        rows = expected_group_rows(plan,b['group_id'],'g0_score')
        check(plain(rows) == plain(b['rows']), 'Control SCREEN row coverage mismatch')
    for ref in list(entries['selections']):
        b = bodies[('selections', ref['sha256'])]
        check(len(b['candidates']) == 4, 'Four selected C blocks required')
        for candidate in b['candidates']:
            cb = add('candidates', candidate)
            check(all(cb[k] == b[k] for k in ('group_id', 'assignment_id')), 'Selection candidate mismatch')
        cs = [bodies[('candidates', r['sha256'])]['C'] for r in b['candidates']]
        check(set(cs) == set(CS), 'Selection C coverage mismatch')
        scores = {str(bodies[('candidates', r['sha256'])]['C']):
                  bodies[('candidates', r['sha256'])]['metrics']['selection_score']
                  for r in b['candidates']}
        check(scores == plain(b['scores']) and plain(b['selected']) in plain(b['candidates']),
              'Selection diagnostic/reference mismatch')
    for ref in entries['candidates']:
        b = bodies[('candidates', ref['sha256'])]
        if b['group_id'] in groups:
            add('statistics', b['statistics'])
    resume_reads = {}  # No source feature payloads are consumed for score recovery.

    waiting = []
    # Reuse sessions across groups for their bounded verified-byte LRU; each
    # session keeps at most one current raw/transformed matrix.
    sessions = {}
    for gid in groups:
        if gid not in identities['statistics']:
            waiting.append(gid)
            emit({'event': 'waiting_for_source_statistics', 'group_id': gid})
            continue
        try:
            needed = [(aid, c) for aid in seeds for c in ((CS[0],) if profile else CS)
                      if (gid, aid, c) not in identities['candidates']]
            if needed:
                if 'source_fit' not in sessions:
                    sessions.update({r: open_numeric_session(plan, plan.root, groups, r)
                                     for r in ('source_fit', 'source_select')})
                for role in ('source_fit', 'source_select'):
                    timed('numeric_read_hash', lambda r=role: sessions[r].read(gid), group_id=gid, role=role)
                ref = identities['statistics'][gid]
                # add(statistics) already established exact compatibility. This
                # shared API must therefore reuse the supplied artifact.
                stats = timed('statistics_reuse', lambda: load_or_publish_statistics(
                    plan, gid, sessions['source_fit'], producer, [ref]), group_id=gid)
                check(plain(stats['reference']) == ref, 'G0 cannot publish statistics')
                for aid, c in needed:
                    ref = timed('fit_compute_write_readback', lambda: fit_candidate_block(
                        plan, gid, aid, c, sessions, stats, producer),
                        group_id=gid, assignment_id=aid, C=c)
                    body = add('candidates', ref)
                    emit({'event': 'candidate_reference', 'reference': plain(ref), 'group_id': gid,
                          'assignment_id': aid, 'C': c,
                          'solver_seconds': sum(a['seconds'] for h in body['records'] for a in h['attempts']),
                          'retry_heads': sum(len(h['attempts']) > 1 for h in body['records'])})
            if not profile:
                for aid in seeds:
                    key = (gid, aid)
                    if key not in identities['selections']:
                        refs = [identities['candidates'][(gid, aid, c)] for c in CS]
                        add('selections', timed('selection', lambda: select_system(
                            plan, gid, aid, refs, producer), group_id=gid, assignment_id=aid))
                    if key not in identities['control_scores']:
                        if 'g0_score' not in sessions:
                            sessions['g0_score'] = open_numeric_session(plan, plan.root, groups, 'g0_score')
                        timed('numeric_read_hash', lambda: sessions['g0_score'].read(gid), group_id=gid, role='g0_score')
                        refs = timed('control_score_write_readback', lambda: score_g0(
                            plan, [identities['selections'][key]], sessions['g0_score'], producer),
                            group_id=gid, assignment_id=aid)
                        check(len(refs) == 1, 'One score bundle per control required')
                        add('control_scores', refs[0])
                triple = [identities['control_scores'].get((gid, aid)) for aid in SEEDS]
                if all(triple) and gid not in identities['null_groups']:
                    result = timed('null_1999_compute_write_readback', lambda: evaluate_g0(
                        plan, triple, entries['assignments'], [], producer), group_id=gid)
                    add('null_groups', result['groups'][gid])
                    if result['gate']:
                        add('gates', result['gate'])
            checkpoint(gid, 'group_checkpoint')
        except (ContractError, OSError, ValueError, KeyError, KeyboardInterrupt) as exc:
            checkpoint(gid, 'failure_checkpoint')
            emit({'event': 'blocked', 'group_id': gid, 'reason': str(exc), 'indexes': indexes})
            raise

    # Include complete triples from distributed seed handoffs, even when those
    # groups are not owned for fitting in this process. Partial triples just wait.
    if not profile:
        for gid in sorted(plan['groups']):
            triple = [identities['control_scores'].get((gid, a)) for a in SEEDS]
            if all(triple):
                resumed = [identities['null_groups'][gid]] if gid in identities['null_groups'] else []
                result = timed('null_group_verify_or_compute', lambda: evaluate_g0(
                    plan, triple, entries['assignments'], resumed, producer), group_id=gid)
                new = gid not in identities['null_groups']
                add('null_groups', result['groups'][gid])
                if result['gate']:
                    add('gates', result['gate'])
                if new:
                    checkpoint(gid, 'null_group_checkpoint')
    complete = set(identities['null_groups']) == set(plan['groups'])
    gate = entries['gates'][0] if entries['gates'] else None
    if complete and not profile:
        if gate:
            body = bodies[('gates', gate['sha256'])]
            check(plain(body['groups']) == identities['null_groups'] and
                  plain(body['assignments']) == plain(assignments), 'Conflicting gate coverage')
        else:
            result = timed('global_reduce_write_readback', lambda: evaluate_g0(
                plan, [], entries['assignments'], entries['null_groups'], producer))
            gate = plain(result['gate'])
            add('gates', gate)
        index = publish_run_index(plan, {'gates': [gate], 'null_groups': entries['null_groups'],
                                        'assignments': entries['assignments']}, producer)
        indexes.append(plain(index))
    result = {'event': 'g0_worker_result', 'stage3_complete': False,
              'coverage_complete': complete, 'gate': gate, 'waiting_statistics': waiting,
              'indexes': indexes, 'entries': entries, 'counts': {k: len(v) for k, v in entries.items()},
              'wall_seconds': time.perf_counter()-started, 'timings': timings,
              'peak_process_bytes': peak_process_bytes(), 'resume_score_payload_reads': resume_reads,
              'session_payload_reads': {r: sum(s.store.reads.values()) for r, s in sessions.items()},
              'session_payload_bytes': {r: sum((plan.root / p).stat().st_size*n
                                              for p, n in s.store.reads.items())
                                        for r, s in sessions.items()},
              'timing_scope': 'wall/process CPU by phase; shared fit/score/null calls include publication/readback'}
    emit(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--settings', required=True, help='Explicit ownership/reference JSON')
    parser.add_argument('--storage-scope', choices=('persistent-drive', 'local-development'), default='persistent-drive')
    args = parser.parse_args()
    from src.common.storage import phase1_root
    root = phase1_root(args.phase1_root)
    protocol = json.loads((Path(__file__).resolve().parents[2]/'configs/protocol.json').read_text(encoding='utf-8'))
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    check({'groups', 'seeds', 'assignments'} <= set(settings), 'Missing ownership/assignment settings')
    check(set(settings) <= {'groups', 'seeds', 'assignments', 'supplied', 'resume_indexes',
                            'statistics_indexes', 'profile'}, 'Unknown G0 settings')
    producer = current_producer(args.storage_scope)
    plan = load_fit_plan(args.manifest, root, protocol)
    result = run_g0(plan, producer=producer, **settings)
    if result['waiting_statistics']:
        raise SystemExit(3)
    if result['gate']:
        b = read_completed(plan, result['gate'], 'g0_gate', {})
        if b['flagged']:
            raise SystemExit(4)


if __name__ == '__main__':
    try:
        main()
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({'event': 'blocked', 'reason': str(exc), 'stage3_complete': False}), flush=True)
        raise SystemExit(2)
