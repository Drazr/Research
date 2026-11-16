"""G0 software fixtures only; no production fits or scientific conclusions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

from test_stage3_foundation import Fixture
from src.common.contracts import ContractError, fingerprint
from src.stage2.schema import plain
from src.stage3.artifacts import Store, write
from src.stage3.data import FitPlan
from src.stage3.interfaces import (open_numeric_session, load_or_publish_statistics,
    publish_target_assignments, publish_run_index, read_completed)
from src.stage3.controls import auc_batch, evaluate_g0
from src.stage3.readouts import targets, donor_map
from src.stage3.run_g0 import run_g0


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(Path(cls.temp.name))
        cls.plan = cls.fixture.build()
        cls.producer = cls.fixture.producer
        cls.assignments = publish_target_assignments(cls.plan, cls.plan.root, cls.producer)
        cls.gids = [gid for gid, g in cls.plan['groups'].items()
                    if g['source'] == 'vlm' and g['language'] == 'en'][:2]
        cls.stats = {}
        for gid in cls.gids:
            session = open_numeric_session(cls.plan, cls.plan.root, [gid], 'source_fit')
            cls.stats[gid] = load_or_publish_statistics(cls.plan, gid, session, cls.producer)['reference']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_worker(self, groups=None, seeds=None, **kwargs):
        return run_g0(self.plan, self.gids[:1] if groups is None else groups,
                      ['seed0'] if seeds is None else seeds, self.assignments,
                      self.producer, **kwargs)

    def test_01_partial_failure_seed_handoffs_null_and_full_resume(self):
        import src.stage3.run_g0 as worker
        gid = self.gids[0]
        events = []
        original = worker.fit_candidate_block
        calls = []
        def interrupted(*args):
            calls.append((args[2], args[3]))
            if len(calls) == 2:
                raise ContractError('fixture interrupted after completed first C')
            return original(*args)
        with patch.object(worker, 'fit_candidate_block', side_effect=interrupted), \
             patch('src.stage3.readouts.statistics', side_effect=AssertionError('G0 must reuse statistics')):
            with self.assertRaisesRegex(ContractError, 'fixture interrupted'):
                self.run_worker(supplied={'statistics': [self.stats[gid]]}, emit=events.append)
        failure = next(e for e in events if e['event'] == 'failure_checkpoint')
        self.assertEqual(failure['counts']['candidates'], 1)
        self.assertTrue(any(e['event'] == 'candidate_reference' for e in events))
        with patch.object(worker, 'fit_candidate_block', wraps=original) as fitter:
            result = self.run_worker(resume_indexes=[failure['index']], emit=lambda e: None)
        self.assertEqual(fitter.call_count, 3)
        self.assertEqual(result['counts']['selections'], 1)
        self.assertEqual(result['counts']['control_scores'], 1)
        self.assertEqual(result['counts']['null_groups'], 0)
        # A separate runtime takes the seed0 index and owns seed1/seed2. The
        # complete triple yields exactly one compact all-1999-draw summary.
        result = self.run_worker(seeds=['seed1', 'seed2'], resume_indexes=result['indexes'], emit=lambda e: None)
        self.assertEqual(result['counts']['candidates'], 12)
        self.assertEqual(result['counts']['selections'], 3)
        self.assertEqual(result['counts']['control_scores'], 3)
        self.assertEqual(result['counts']['null_groups'], 1)
        self.assertFalse(result['coverage_complete'])
        self.assertIsNone(result['gate'])
        ref = result['entries']['null_groups'][0]
        store = Store(self.plan.root, self.plan.protocol)
        self.assertEqual(set(store.complete(ref)['files']), {'body.json', 'maxima.npy'})
        self.assertEqual(store.array(ref, 'maxima.npy', '<f8', (1999,)).shape, (1999,))
        with patch.object(worker, 'fit_candidate_block', side_effect=AssertionError('completed fit repeated')), \
             patch.object(worker, 'score_g0', side_effect=AssertionError('completed score repeated')), \
             patch('src.stage3.controls._null_group', side_effect=AssertionError('completed null repeated')):
            resumed = self.run_worker(seeds=list(worker.SEEDS), resume_indexes=result['indexes'], emit=lambda e: None)
            # Null-only recovery closes over scores, selections and all Cs.
            reduced = self.run_worker(groups=[], seeds=[], supplied={'null_groups': [ref]}, emit=lambda e: None)
        self.assertEqual(resumed['counts'], result['counts'])
        self.assertEqual(reduced['counts']['control_scores'], 3)
        self.assertEqual(reduced['counts']['candidates'], 12)
        self.assertFalse(resumed['stage3_complete'])
        self.assertGreater(result['wall_seconds'], 0)
        self.assertTrue(result['session_payload_reads'])
        self.assertTrue(all(t['wall_seconds'] >= 0 for t in result['timings']))
        scores = result['entries']['control_scores']
        with self.assertRaises(ContractError):
            evaluate_g0(self.plan, scores[:2], [self.assignments], [], self.producer)
        sb = store.json(scores[0], 'body.json')
        arrays = {k: store.array(scores[0], k+'.npy') for k in ('logits', 'estimable')}
        duplicate = write(self.plan, 'fixture/duplicate-score', 'control_scores',
                          {k: sb[k] for k in ('selection', 'group_id', 'assignment_id', 'rows', 'metrics')},
                          arrays, self.producer)
        with self.assertRaisesRegex(ContractError, 'Conflicting control_scores'):
            self.run_worker(groups=[], seeds=[], supplied={'control_scores': [scores[0], duplicate]}, emit=lambda e: None)
        with self.assertRaisesRegex(ContractError, 'Conflicting resumed score'):
            evaluate_g0(self.plan, [duplicate]+scores[1:], [self.assignments], [ref], self.producer)
        bad_rows = dict(sb, rows=sb['rows'][:-1])
        bad = write(self.plan, 'fixture/wrong-rows', 'control_scores',
                    {k: bad_rows[k] for k in ('selection', 'group_id', 'assignment_id', 'rows', 'metrics')},
                    arrays, self.producer)
        with self.assertRaisesRegex(ContractError, 'SCREEN row coverage'):
            self.run_worker(groups=[], seeds=[], supplied={'control_scores': [bad]}, emit=lambda e: None)
        nb = store.json(ref, 'body.json')
        forged = write(self.plan, 'fixture/duplicate-null-seeds', 'null_group',
                       {k: ([scores[0]]*3 if k == 'scores' else nb[k])
                        for k in ('group_id', 'scores', 'draw_ids', 'observed', 'family_identity')},
                       {'maxima': store.array(ref, 'maxima.npy')}, self.producer)
        with self.assertRaisesRegex(ContractError, 'Duplicate null score seed'):
            self.run_worker(groups=[], seeds=[], supplied={'null_groups': [forged]}, emit=lambda e: None)
        # Fresh-process semantic candidate readback, no source matrix fitting.
        import subprocess, sys, os
        payload = {'root': str(self.plan.root), 'plan': plain(self.plan.reference),
                   'protocol': self.plan.protocol, 'candidate': result['entries']['candidates'][0],
                   'scores': result['entries']['control_scores'], 'null': plain(ref)}
        script = ('import json,sys,numpy; from sklearn.metrics import roc_auc_score; '
                  'from src.stage3.interfaces import load_fit_plan,read_completed; '
                  'd=json.load(sys.stdin); p=load_fit_plan(d["plan"],d["root"],d["protocol"]); '
                  'read_completed(p,d["candidate"],"candidate",{"assignment_id":"seed0"}); '
                  '[read_completed(p,r,"control_scores",{}) for r in d["scores"]]; '
                  'read_completed(p,d["null"],"null_group",{}); print("verified")')
        child = subprocess.run([sys.executable, '-c', script], input=json.dumps(payload),
                               text=True, capture_output=True, cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
        self.assertEqual(child.returncode, 0, child.stdout+child.stderr)
        self.assertIn('verified', child.stdout)
        print(json.dumps({'fixture_g0_counts': result['counts'], 'wall_seconds': result['wall_seconds'],
                          'payload_reads': result['session_payload_reads'],
                          'payload_bytes': result['session_payload_bytes']}), flush=True)

    def test_missing_group_statistics_does_not_block_ready_group(self):
        ready, missing = self.gids
        ordinary = publish_run_index(self.plan, {'statistics': [self.stats[ready]]}, self.producer)
        with patch('src.stage3.readouts.statistics', side_effect=AssertionError('G0 scaler publication')):
            result = self.run_worker(groups=[missing, ready], statistics_indexes=[ordinary], emit=lambda e: None)
        self.assertEqual(result['waiting_statistics'], [missing])
        self.assertEqual(result['counts']['candidates'], 4)
        self.assertEqual(result['counts']['control_scores'], 1)
        self.assertEqual({read_completed(self.plan, r, 'candidate', {})['group_id']
                          for r in result['entries']['candidates']}, {ready})

    def test_ownership_assignment_and_conflict_guards(self):
        for groups, seeds in ((self.gids[:1]*2, ['seed0']), (['foreign'], ['seed0']),
                              (self.gids[:1], ['true']), (self.gids[:1], ['seed3']),
                              (self.gids[:1], ['seed0', 'seed0'])):
            with self.assertRaises(ContractError):
                self.run_worker(groups=groups, seeds=seeds, emit=lambda e: None)

    def test_wrong_assignments_and_unknown_resume_category(self):
        bad = write(self.plan, 'fixture/bad', 'assignments', {'maps': {}, 'assignment_identity': '0'*64}, {}, self.producer)
        with patch('src.stage3.run_g0.fit_candidate_block', side_effect=AssertionError('must reject before fit')):
            with self.assertRaises(ContractError):
                run_g0(self.plan, self.gids[:1], ['seed0'], bad, self.producer, emit=lambda e: None)
        with self.assertRaises(ContractError):
            self.run_worker(supplied={'latest': []}, emit=lambda e: None)

    def test_corrupt_and_conflicting_statistics_rejected_before_fit(self):
        gid = self.gids[0]
        b = read_completed(self.plan, self.stats[gid], 'statistics', {'group_id': gid})
        store = Store(self.plan.root, self.plan.protocol)
        arrays = {k: store.array(self.stats[gid], k+'.npy') for k in ('mean', 'variance', 'scale')}
        duplicate = write(self.plan, 'fixture/duplicate', 'statistics',
                          {k: plain(b[k]) for k in ('group_id', 'input_identity', 'center')}, arrays, self.producer)
        with self.assertRaisesRegex(ContractError, 'Conflicting statistics'):
            self.run_worker(supplied={'statistics': [self.stats[gid], duplicate]}, emit=lambda e: None)
        path = (self.plan.root/duplicate['path']).with_name('scale.npy')
        path.write_bytes(b'partial/corrupt')
        with self.assertRaises(ContractError):
            self.run_worker(supplied={'statistics': [duplicate]}, emit=lambda e: None)

    def test_roles_on_cached_numeric_session(self):
        gid = self.gids[0]
        s = open_numeric_session(self.plan, self.plan.root, [gid], 'source_fit')
        s.read(gid)
        with self.assertRaises(ContractError): s.read(gid, 'g0_score')
        with self.assertRaises(ContractError): s.read(self.gids[1])
        for role in ('CONFIRM', 'screen', 'source_score', 'DEV-C'):
            with self.assertRaises(ContractError):
                open_numeric_session(self.plan, self.plan.root, [gid], role)

    def test_representative_profile_and_ordinary_screen_rejection(self):
        from src.stage3.interfaces import fit_candidate_block, select_system, load_fitted_system
        gid = self.plan['profiling_groups'][0]
        sessions = {r: open_numeric_session(self.plan, self.plan.root, [gid], r)
                    for r in ('source_fit', 'source_select')}
        stats = load_or_publish_statistics(self.plan, gid, sessions['source_fit'], self.producer)
        result = self.run_worker(groups=[gid], profile=True,
                                 supplied={'statistics': [stats['reference']]}, emit=lambda e: None)
        self.assertEqual(result['counts']['candidates'], 1)
        self.assertEqual(result['counts']['selections'], 0)
        self.assertEqual(result['counts']['control_scores'], 0)
        self.assertNotIn('g0_score', result['session_payload_reads'])
        with self.assertRaises(ContractError):
            self.run_worker(groups=[gid], seeds=['seed1'], profile=True, emit=lambda e: None)
        candidates = [fit_candidate_block(self.plan, gid, 'true', c, sessions, stats, self.producer)
                      for c in (.01, .1, 1., 10.)]
        selected = select_system(self.plan, gid, 'true', candidates, self.producer)
        with self.assertRaisesRegex(ContractError, 'Ordinary SCREEN'):
            load_fitted_system(selected, self.plan.root, self.plan, 'g0_score',
                               {'group_id': gid, 'assignment_id': 'true'})
        with self.assertRaisesRegex(ContractError, 'control targets'):
            self.run_worker(groups=[], seeds=[], supplied={'selections': [selected]}, emit=lambda e: None)


class DiagnosticTests(unittest.TestCase):
    def test_shared_whole_vectors_and_moving_primary_unknown_masks(self):
        # Two primary cells, including tied scores and an unknown donor cell.
        targets_by_id = {str(i): {'split': 'SCREEN', 'values': [i % 2, (i//2) % 2]}
                         for i in range(8)}
        targets_by_id['0']['values'][0] = None
        plan = {'targets': targets_by_id}
        rows = [{'image_id': str(i), 'split': 'SCREEN', 'language': lang}
                for lang in ('en', 'hi', 'bn', 'te') for i in range(8)]
        scores = np.array([[i % 3, i % 4] for i in range(8)], dtype=float)
        original = copy.deepcopy(plan)
        positions = set()
        for aid in ('seed0', 'seed1', 'seed2'):
            for draw in (None, 1, 2, 13, 100, 1999):
                y, known = targets(plan, rows, aid, draw)
                for start in (8, 16, 24):
                    np.testing.assert_array_equal(y[:8], y[start:start+8])
                    np.testing.assert_array_equal(known[:8], known[start:start+8])
                positions.add(tuple(np.flatnonzero(~known[:8, 0])))
                got = auc_batch(scores, y[None, :8], known[None, :8])[0]
                expected = [roc_auc_score(y[:8, k][known[:8, k]], scores[:, k][known[:8, k]])
                            for k in range(2)]
                np.testing.assert_allclose(got, expected, atol=1e-14)
                mapping = donor_map(plan, aid, 'SCREEN', draw)
                for index, row in enumerate(rows[:8]):
                    donor = targets_by_id[mapping[row['image_id']]]['values']
                    self.assertEqual(known[index].tolist(), [v is not None for v in donor])
        self.assertGreater(len(positions), 1)
        self.assertEqual(plan, original)
        with self.assertRaises(ContractError): donor_map(plan, 'seed0', 'CONFIRM')

    def test_group_max_reduction_equivalent_and_coverage_guards(self):
        # Controlled synthetic all-draw arrays isolate the reducer, not fitting.
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp)); original = f.build()
            body = plain(original); body['groups'] = {g: body['groups'][g] for g in list(body['groups'])[:2]}
            p = FitPlan(body, original.root, original.protocol, original.reference)
            assignment = publish_target_assignments(p, p.root, f.producer)
            rng = np.random.default_rng(9)
            family = rng.uniform(0, .49, (2, 3, 1999, 4))
            observed_family = rng.uniform(0, .49, (2, 3, 4))
            refs = []
            for index, gid in enumerate(p['groups']):
                refs.append(write(p, 'fixture/nulls', 'null_group',
                    {'group_id': gid, 'scores': [], 'draw_ids': list(range(1, 2000)),
                     'observed': float(observed_family[index].max()),
                     'family_identity': fingerprint({'groups': plain(p['groups']), 'primary': plain(p['primary']), 'draws': 1999})},
                    {'maxima': family[index].max(axis=(0, 2))}, f.producer))
            partial = evaluate_g0(p, [], [assignment], refs[:1], f.producer)
            self.assertFalse(partial['complete']); self.assertIsNone(partial['gate'])
            result = evaluate_g0(p, [], [assignment], refs, f.producer)
            store = Store(p.root, p.protocol)
            expected = family.max(axis=(0, 1, 3))
            np.testing.assert_array_equal(store.array(result['gate'], 'maxima.npy'), expected)
            gate = store.json(result['gate'], 'body.json')
            observed = float(observed_family.max())
            self.assertEqual(gate['observed'], observed)
            self.assertEqual(gate['p'], (1+int((expected >= observed).sum()))/2000)
            for bad in (refs[:1]*2, refs+refs[:1]):
                with self.assertRaises(ContractError): evaluate_g0(p, [], [assignment], bad, f.producer)


if __name__ == '__main__':
    unittest.main()
