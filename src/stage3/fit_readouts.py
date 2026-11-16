"""Ordinary FIT/DEV-S scheduling over the frozen coordinator interfaces.

One invocation owns an explicit partition of groups. No solver, scaler, raw
feature producer, ordinary evaluation or Stage 3 acceptance lives here.
"""
import os

# These must precede even the transitive numerical imports below.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import threading
import time

from src.common.contracts import ContractError
from src.common.storage import phase1_root
from src.stage2.schema import check, plain
from src.stage3.interfaces import (
    CS, current_producer, load_fit_plan, open_numeric_session,
    load_or_publish_statistics, fit_candidate_block, load_candidate_block,
    select_system, read_completed, publish_run_index, load_run_index,
)


def emit(event, **fields):
    print(json.dumps(dict(event=event, **plain(fields)), allow_nan=False), flush=True)


def peak_memory():
    """Actual process high-water RSS on Colab/Linux; never invent a Windows value."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except ImportError:
        return None


def measured(event, operation, report=emit, **fields):
    """Stream heartbeats during a complete block; profile without altering APIs.

    The cProfile exclusive I/O/hash times are components, not additive elapsed
    phase estimates. Publication and semantic readback are separately timed by
    their public calls. No monkey-patching or new persistent timing cache.
    """
    start = time.monotonic()
    report(event + '_start', **fields)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(15):
            report('heartbeat', phase=event, elapsed_seconds=time.monotonic()-start,
                   peak_rss_bytes=peak_memory(), **fields)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    profiler = cProfile.Profile()
    try:
        return profiler.runcall(operation)
    finally:
        stop.set()
        thread.join()
        stats = pstats.Stats(profiler).stats
        # Exclusive time avoids double counting Path.read_bytes and file.read.
        times = {'read': 0., 'hash': 0., 'write': 0.}
        for (_, _, name), (_, _, exclusive, _, _) in stats.items():
            if 'hashlib' in name or '_hashlib' in name:
                times['hash'] += exclusive
            elif "'read' of '_io." in name or "'readinto' of '_io." in name:
                times['read'] += exclusive
            elif "'write' of '_io." in name:
                times['write'] += exclusive
        report(event + '_end', elapsed_seconds=time.monotonic()-start,
               io_hash_exclusive_seconds=times, peak_rss_bytes=peak_memory(), **fields)


def owned_groups(plan, ownership, owner, feature_source, model=None):
    """Validate a complete, disjoint 145-group assignment before any fits/IO."""
    check(feature_source in ('vlm', 'e03'), 'FEATURE_SOURCE must be vlm/e03')
    check((feature_source == 'vlm' and model in ('qwen', 'gemma')) or
          (feature_source == 'e03' and model is None), 'MODEL applies only to VLM')
    check(isinstance(ownership, dict) and set(ownership) == {'owners'}, 'Explicit owners mapping required')
    owners = ownership['owners']
    check(isinstance(owners, dict) and owner in owners, 'Unknown owner')
    all_groups = []
    for name, gids in owners.items():
        check(isinstance(name, str) and name and isinstance(gids, list) and gids,
              'Each owner needs an explicit nonempty group list')
        check(all(isinstance(g, str) for g in gids), 'Group IDs must be strings')
        all_groups.extend(gids)
    check(len(plan['groups']) == 145 and len(plan['categories']) == 25 and len(plan['primary']) == 14,
          'Unexpected ordinary inventory/cohort')
    check(len(all_groups) == len(set(all_groups)) and set(all_groups) == set(plan['groups']),
          'Ownership must cover all 145 groups exactly once')
    selected = owners[owner]
    check(all(plan['groups'][g]['source'] == feature_source and
              (feature_source == 'e03' or plan['groups'][g]['model'] == model) for g in selected),
          'Owner partition disagrees with FEATURE_SOURCE/MODEL')
    return tuple(sorted(selected))


class Recovery:
    """Semantic validation of explicit owned units, indexed once per invocation."""
    def __init__(self, plan, groups, indexes, profile_indexes=()):
        self.plan, self.groups = plan, set(groups)
        self.statistics, self.candidates, self.selections = {}, {}, {}
        seen = set()
        sources = [(ref, False) for ref in indexes] + [(ref, True) for ref in profile_indexes]
        for index, shared_profile in sources:
            entries = load_run_index(plan, index)
            check(shared_profile or set(entries) <= {'statistics', 'candidates', 'selections'},
                  'Use ordinary/profile indexes, not G0 score/null/assignment indexes')
            for name, refs in entries.items():
                if name not in ('statistics', 'candidates', 'selections'):
                    continue
                for ref in refs:
                    token = (name, ref['path'], ref['sha256'])
                    if token in seen:
                        continue  # Overlapping snapshots of the same immutable unit.
                    seen.add(token)
                    kind = {'statistics': 'statistics', 'candidates': 'candidate', 'selections': 'selection'}[name]
                    # Statistics need an explicit group for the semantic loader.
                    # The index contains no group keys; read only its authenticated
                    # small body first, then the public semantic loader below.
                    body = plan.store.json(ref, 'body.json', 'stage3_' + kind)
                    gid = body['group_id']
                    if shared_profile and (gid not in self.groups or
                            (kind != 'statistics' and body['assignment_id'] != 'true')):
                        continue
                    check(gid in self.groups, 'Resume reference outside assigned groups')
                    expected = {'group_id': gid}
                    if kind != 'statistics':
                        aid = body['assignment_id']
                        check(aid == 'true', 'Non-ordinary resume unit; supply G0 evidence as profile indexes')
                        expected['assignment_id'] = aid
                    body = read_completed(plan, ref, kind, expected)
                    if kind == 'statistics':
                        key, table = gid, self.statistics
                    elif kind == 'candidate':
                        key, table = (gid, body['assignment_id'], body['C']), self.candidates
                    else:
                        key, table = gid, self.selections
                    check(key not in table, 'Conflicting duplicate completed coverage')
                    table[key] = plain(ref)
        # Selections are never trusted as substitutes for the four original Cs.
        for gid, ref in self.selections.items():
            body = read_completed(plan, ref, 'selection', {'group_id': gid, 'assignment_id': 'true'})
            refs = [self.candidates.get((gid, 'true', c)) for c in CS]
            check(list(body['candidates']) == refs,
                  'Selection requires the same four explicit candidate references')
            scores = {str(c): plan.store.json(candidate, 'body.json', 'stage3_candidate')['metrics']['selection_score']
                      for c, candidate in zip(CS, refs)}
            check(plain(body['scores']) == scores and
                  body['selected'] == self.candidates[(gid, 'true', body['selected_C'])],
                  'Selection scores/reference disagree with original candidates')
        for (gid, _, _), ref in self.candidates.items():
            body = plan.store.json(ref, 'body.json', 'stage3_candidate')
            check(body['statistics'] == self.statistics.get(gid), 'Candidate requires its owned statistics reference')

    def entries(self, gid):
        return {'statistics': [self.statistics[gid]] if gid in self.statistics else [],
                'candidates': [ref for (group, _, _), ref in sorted(self.candidates.items()) if group == gid],
                'selections': [self.selections[gid]] if gid in self.selections else []}


def require_profiles(plan, indexes):
    """All metadata-selected production representatives precede bulk work."""
    required = {(g, aid, c) for g in plan['profiling_groups']
                for aid, c in (('true', CS[0]), ('true', CS[-1]), ('seed0', CS[0]))}
    found, statistics = {}, {}
    for index in indexes:
        for ref in load_run_index(plan, index).get('candidates', ()):
            body = plan.store.json(ref, 'body.json', 'stage3_candidate')
            key = (body['group_id'], body['assignment_id'], body['C'])
            if key not in required:
                continue
            check(key not in found or found[key] == plain(ref), 'Conflicting profiling coverage')
            if key not in found:
                read_completed(plan, ref, 'candidate', {'group_id': key[0], 'assignment_id': key[1], 'C': key[2]})
                check(key[0] not in statistics or statistics[key[0]] == body['statistics'],
                      'Representatives must reuse the ordinary-owned source statistics')
                statistics[key[0]] = body['statistics']
                found[key] = plain(ref)
    check(set(found) == required, 'Bulk requires all representative true .01/10 and seed0 .01 references')


def run(plan, ownership, owner, feature_source, producer, model=None,
        resume_indexes=(), profile_indexes=(), mode='ordinary', report=emit):
    groups = owned_groups(plan, ownership, owner, feature_source, model)
    check(mode in ('profile', 'ordinary'), 'Unknown worker mode')
    if mode == 'ordinary':
        require_profiles(plan, profile_indexes)
    else:
        groups = tuple(g for g in groups if g in plan['profiling_groups'])
        check(groups, 'This owner has no metadata-selected representatives')
    recovery = Recovery(plan, groups, resume_indexes, profile_indexes)
    report('worker_start', owner=owner, feature_source=feature_source, model=model,
           mode=mode, groups=list(groups), plan=plain(plan.reference),
           code_commit=producer['code_commit'], runtime=plain(producer['software_versions']),
           storage_scope=producer['storage_scope'], fit_identity=plan['fit_identity'],
           stage3_complete=False)
    # Two bounded byte caches are reused across groups/sites/languages. Each
    # retains only its current raw/transformed matrix, never all 145 matrices.
    sessions = None
    indexes = []
    for gid in groups:
        dirty = False
        fields = {'group_id': gid, 'owner': owner}
        # G0 owns seed0 .01 representatives and supplies their immutable indexes
        # to the bulk gate. Ordinary never competes with that control producer.
        tasks = [('true', c) for c in CS] if mode == 'ordinary' else [
            ('true', CS[0]), ('true', CS[-1])]
        missing = [(aid, c) for aid, c in tasks if (gid, aid, c) not in recovery.candidates]
        try:
            if missing or gid not in recovery.statistics:
                if sessions is None:
                    sessions = {role: open_numeric_session(plan, plan.root, groups, role)
                                for role in ('source_fit', 'source_select')}
                rows, matrix = measured('numeric_read_verify', lambda: sessions['source_fit'].read(gid), report, **fields)
                report('source_metadata', shape=list(matrix.shape), rows=len(rows),
                       nnz=int(matrix.nnz) if hasattr(matrix, 'nnz') else None,
                       source=plain(plan['groups'][gid]), **fields)
                stats = measured('statistics_load_publish', lambda: load_or_publish_statistics(
                    plan, gid, sessions['source_fit'], producer,
                    [recovery.statistics[gid]] if gid in recovery.statistics else ()), report, **fields)
                if gid not in recovery.statistics:
                    recovery.statistics[gid] = plain(stats['reference'])
                    dirty = True
                report('statistics_ready', reference=plain(stats['reference']), reuse_for='G0', **fields)
                measured('selection_numeric_read_verify', lambda: sessions['source_select'].read(gid), report, **fields)
                del matrix, rows
            for aid, c in tasks:
                key = (gid, aid, c)
                block_fields = dict(fields, assignment_id=aid, C=c)
                if key in recovery.candidates:
                    report('candidate_reused', reference=recovery.candidates[key], **block_fields)
                    continue
                ref = measured('fit_publish', lambda: fit_candidate_block(
                    plan, gid, aid, c, sessions, stats, producer), report, **block_fields)
                # Record the completed reference before any further reporting or
                # readback can fail, so handled failures preserve useful work.
                recovery.candidates[key] = plain(ref)
                dirty = True
                report('candidate_published', reference=plain(ref), **block_fields)
                body, coef, intercepts, valid = measured('candidate_readback',
                    lambda: load_candidate_block(plan, ref), report, **block_fields)
                report('candidate_complete', reference=plain(ref), statuses=[r['status'] for r in body['records']],
                       compute_fit_seconds=sum(a['seconds'] for r in body['records'] for a in r['attempts']),
                       retries=sum(len(r['attempts']) > 1 for r in body['records']), **block_fields)
                del body, coef, intercepts, valid
            if mode == 'ordinary' and gid not in recovery.selections:
                refs = [recovery.candidates[(gid, 'true', c)] for c in CS]
                ref = measured('selection_publish', lambda: select_system(plan, gid, 'true', refs, producer), report, **fields)
                recovery.selections[gid] = plain(ref)
                dirty = True
                report('selection_ready', reference=plain(ref), **fields)
            # Small per-group indexes avoid repeatedly writing a growing job inventory.
            ref = publish_run_index(plan, recovery.entries(gid), producer)
            indexes.append(plain(ref))
            report('group_complete', index=plain(ref), counts={k: len(v) for k, v in recovery.entries(gid).items()},
                   mode=mode, stage3_complete=False, **fields)
        except (Exception, KeyboardInterrupt) as exc:
            if dirty:
                try:
                    ref = publish_run_index(plan, recovery.entries(gid), producer)
                    report('recovery_index', index=plain(ref), **fields)
                except Exception as index_error:
                    report('recovery_index_failed', reason=str(index_error),
                           completed=recovery.entries(gid), **fields)
            report('worker_blocked', reason=str(exc), stage3_complete=False, **fields)
            raise
    result = {'owner': owner, 'mode': mode, 'groups': len(groups), 'indexes': indexes,
              'stage3_complete': False}
    report('worker_complete', **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-root', required=True)
    parser.add_argument('--manifest', required=True, help='Explicit prepared Stage 3 plan completion marker')
    parser.add_argument('--ownership', required=True, help='JSON owners mapping covering all 145 groups')
    parser.add_argument('--owner', required=True)
    parser.add_argument('--feature-source', choices=('vlm', 'e03'), required=True)
    parser.add_argument('--model', choices=('qwen', 'gemma'))
    parser.add_argument('--mode', choices=('profile', 'ordinary'), default='profile')
    parser.add_argument('--resume-index', action='append', default=[], help='Immutable {path,sha256} JSON')
    parser.add_argument('--profile-index', action='append', default=[], help='Immutable representative index JSON')
    parser.add_argument('--storage-scope', choices=('local-development', 'persistent-drive'), default='persistent-drive')
    a = parser.parse_args()
    check(sys.version_info[:2] in ((3, 11), (3, 12)), 'Use Python 3.11/3.12')
    root = phase1_root(a.phase1_root)
    protocol = json.loads((Path(__file__).resolve().parents[2] / 'configs/protocol.json').read_text(encoding='utf-8'))
    producer = current_producer(a.storage_scope)
    plan = load_fit_plan(a.manifest, root, protocol)
    run(plan, json.loads(Path(a.ownership).read_text(encoding='utf-8')), a.owner,
        a.feature_source, producer, a.model, [json.loads(r) for r in a.resume_index],
        [json.loads(r) for r in a.profile_index], a.mode)


if __name__ == '__main__':
    try:
        main()
    except (ContractError, OSError, KeyError, ValueError) as exc:
        emit('worker_blocked', reason=str(exc), stage3_complete=False)
        raise SystemExit(2)
