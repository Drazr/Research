"""Ordinary orchestration fixtures. These are not production fit evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError
from src.stage2.schema import plain
from src.stage3.interfaces import CS, load_run_index, read_completed, fit_candidate_block, publish_run_index
from src.stage3.fit_readouts import Recovery, owned_groups, require_profiles, run, measured
from test_stage3_foundation import Fixture


def ownership(plan, chosen=None):
    owners = {'qwen': [], 'gemma': [], 'e03': []}
    for gid, group in plan['groups'].items():
        owners[group['model'] if group['source'] == 'vlm' else 'e03'].append(gid)
    if chosen:
        for gid in chosen:
            for groups in owners.values():
                if gid in groups:
                    groups.remove(gid)
        owners['mine'] = list(chosen)
    return {'owners': {k: v for k, v in owners.items() if v}}


def quiet_measure(event, operation, report=None, **fields):
    return operation()


class OrdinaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temp.name))
        self.plan = self.fixture.build()
        self.producer = self.fixture.producer
        self.events = []

    def tearDown(self):
        self.temp.cleanup()

    def report(self, event, **fields):
        self.events.append(dict(event=event, **fields))

    def execute(self, gids, source='vlm', model='qwen', **kwargs):
        with patch('src.stage3.fit_readouts.measured', side_effect=quiet_measure):
            return run(self.plan, ownership(self.plan, gids), 'mine', source,
                       self.producer, model, report=self.report, **kwargs)

    def test_complete_disjoint_inventory_and_selectors(self):
        owners = ownership(self.plan)
        self.assertEqual([len(owned_groups(self.plan, owners, o, 'e03' if o == 'e03' else 'vlm',
                                         None if o == 'e03' else o)) for o in ('qwen', 'gemma', 'e03')], [66, 66, 13])
        bad = copy.deepcopy(owners)
        bad['owners']['e03'].append(bad['owners']['qwen'][0])
        for value in (bad, {'owners': {'qwen': owners['owners']['qwen']}}):
            with self.assertRaises(ContractError):
                owned_groups(self.plan, value, 'qwen', 'vlm', 'qwen')
        for source, model in [('e03', 'qwen'), ('vlm', None), ('vlm', 'gemma')]:
            with self.assertRaises(ContractError):
                owned_groups(self.plan, owners, 'qwen', source, model)

    def test_profile_true_extremes_handoff_and_g0_bulk_gate(self):
        gid = next(g for g in self.plan['profiling_groups'] if self.plan['groups'][g].get('model') == 'qwen')
        result = self.execute([gid], mode='profile')
        entries = load_run_index(self.plan, result['indexes'][0])
        self.assertEqual({read_completed(self.plan, r, 'candidate', {'group_id': gid})['C']
                          for r in entries['candidates']}, {CS[0], CS[-1]})
        self.assertEqual(len(entries['statistics']), 1)
        self.assertEqual(len(entries['selections']), 0)
        self.assertTrue(any(e['event'] == 'statistics_ready' and e['reuse_for'] == 'G0' for e in self.events))
        with self.assertRaises(ContractError):
            require_profiles(self.plan, result['indexes'])
        with patch('src.stage3.fit_readouts.fit_candidate_block', side_effect=AssertionError('gate before fitting')):
            with self.assertRaises(ContractError):
                self.execute([gid], mode='ordinary', resume_indexes=result['indexes'])
        # Profile references alone also serve as recovery: never rerun either
        # measured true block merely because it wasn't repeated in --resume-index.
        with patch('src.stage3.fit_readouts.require_profiles'), \
             patch('src.stage3.fit_readouts.fit_candidate_block', wraps=fit_candidate_block) as fitter:
            self.execute([gid], mode='ordinary', profile_indexes=result['indexes'])
        self.assertEqual([c.args[3] for c in fitter.call_args_list], [.1, 1.])
        # The global gate requires all six representatives and their seed0
        # blocks; mock only the manifest plumbing for complete-family coverage.
        bodies = {}
        for group in self.plan['profiling_groups']:
            for aid, c in [('true', .01), ('true', 10.), ('seed0', .01)]:
                token = f'{group}/{aid}/{c}'
                bodies[token] = {'group_id': group, 'assignment_id': aid, 'C': c,
                                 'statistics': {'path': group, 'sha256': 'a'*64}}
        refs = [{'path': token, 'sha256': 'b'*64} for token in bodies]
        with patch('src.stage3.fit_readouts.load_run_index', return_value={'candidates': refs}), \
             patch.object(self.plan.store, 'json', side_effect=lambda ref, *args: bodies[ref['path']]), \
             patch('src.stage3.fit_readouts.read_completed'):
            require_profiles(self.plan, [{}])
            bodies[refs[-1]['path']]['statistics'] = {'path': 'wrong', 'sha256': 'c'*64}
            with self.assertRaisesRegex(ContractError, 'source statistics'):
                require_profiles(self.plan, [{}])
        self.assertFalse(result['stage3_complete'])

    def test_real_dense_sparse_length_pooled_resume_without_refit_or_input_reads(self):
        # All numerical work uses the coordinator engine on tiny Stage 2 wire fixtures.
        examples = [next(g for g, b in self.plan['groups'].items() if b.get('model') == 'qwen')]
        for family, pooled in [('word_tfidf', False), ('length', False), ('pooled_char_tfidf', True)]:
            examples.append(next(g for g, b in self.plan['groups'].items() if b['source'] == 'e03' and
                                 b['recipe']['family'] == family and (b['language'] is None) == pooled))
        for gid in examples:
            group = self.plan['groups'][gid]
            source, model = group['source'], group.get('model')
            # Representative gate itself is tested separately; these four cases
            # test downstream block/selection orchestration, not global profiling.
            with patch('src.stage3.fit_readouts.require_profiles'):
                result = self.execute([gid], source, model, mode='ordinary')
            entries = load_run_index(self.plan, result['indexes'][0])
            self.assertEqual({k: len(v) for k, v in entries.items()},
                             {'statistics': 1, 'candidates': 4, 'selections': 1})
            for ref in entries['candidates']:
                body = read_completed(self.plan, ref, 'candidate', {'group_id': gid, 'assignment_id': 'true'})
                self.assertEqual(len(body['records']), 25)
                self.assertTrue(all(r['status'] == 'fitted' for r in body['records']))
            selected = read_completed(self.plan, entries['selections'][0], 'selection',
                                      {'group_id': gid, 'assignment_id': 'true'})
            self.assertEqual(selected['selected_C'], CS[0])  # fixture ties choose smallest C
            incomplete = publish_run_index(self.plan, {'statistics': entries['statistics'],
                'candidates': entries['candidates'][:-1], 'selections': entries['selections']}, self.producer)
            with self.assertRaisesRegex(ContractError, 'four explicit candidate'):
                Recovery(self.plan, [gid], [incomplete])
            # A hash-valid selection with altered nonselected scores must not
            # bypass agreement with the four original DEV-S candidate metrics.
            from src.stage3.artifacts import write
            forged = plain(selected)
            forged['scores'][str(CS[-1])] = .25
            false_selection = write(self.plan, 'fixture/forged-selection', 'selection', forged, {}, self.producer)
            false_index = publish_run_index(self.plan, dict(plain(entries), selections=[false_selection]), self.producer)
            with self.assertRaisesRegex(ContractError, 'scores/reference'):
                Recovery(self.plan, [gid], [false_index])
            with patch('src.stage3.fit_readouts.require_profiles'), \
                 patch('src.stage3.fit_readouts.fit_candidate_block', side_effect=AssertionError('no refit')), \
                 patch('src.stage3.fit_readouts.open_numeric_session', side_effect=AssertionError('no source arrays')):
                again = self.execute([gid], source, model, mode='ordinary', resume_indexes=result['indexes'] * 2)
            self.assertEqual(load_run_index(self.plan, again['indexes'][0]), entries)

    def test_failure_indexes_completed_c_and_resumes_missing_only(self):
        gid = next(g for g, b in self.plan['groups'].items() if b.get('model') == 'qwen')
        calls = []

        def fail(plan, group, aid, c, *args):
            calls.append(c)
            if c == .1:
                raise ContractError('fixture interruption')
            return fit_candidate_block(plan, group, aid, c, *args)

        with patch('src.stage3.fit_readouts.require_profiles'), patch('src.stage3.fit_readouts.fit_candidate_block', side_effect=fail):
            with self.assertRaisesRegex(ContractError, 'fixture interruption'):
                self.execute([gid], mode='ordinary')
        index = next(e['index'] for e in self.events if e['event'] == 'recovery_index')
        self.assertEqual(len(load_run_index(self.plan, index)['candidates']), 1)
        self.assertEqual(calls, [.01, .1])
        with patch('src.stage3.fit_readouts.require_profiles'), \
             patch('src.stage3.fit_readouts.fit_candidate_block', wraps=fit_candidate_block) as fitter:
            result = self.execute([gid], mode='ordinary', resume_indexes=[index])
        self.assertEqual([c.args[3] for c in fitter.call_args_list], [.1, 1., 10.])
        self.assertEqual(len(load_run_index(self.plan, result['indexes'][0])['selections']), 1)

    def test_resume_rejects_conflicts_corruption_wrong_owner_and_incomplete_selection(self):
        gid = next(g for g in self.plan['profiling_groups'] if self.plan['groups'][g].get('model') == 'qwen')
        first = self.execute([gid], mode='profile')['indexes'][0]
        second = self.execute([gid], mode='profile')['indexes'][0]
        with self.assertRaisesRegex(ContractError, 'Conflicting duplicate'):
            Recovery(self.plan, [gid], [first, second])
        other = next(g for g in self.plan['groups'] if g != gid)
        with self.assertRaisesRegex(ContractError, 'outside assigned'):
            Recovery(self.plan, [other], [first])
        candidate = load_run_index(self.plan, first)['candidates'][0]
        (self.plan.root / candidate['path']).with_name('coefficients.npy').write_bytes(b'corrupt')
        with self.assertRaises(ContractError):
            Recovery(self.plan, [gid], [first])

    def test_all_145_groups_four_c_complete_scheduling(self):
        # Structural scheduling coverage, explicitly mocked (no scientific results).
        total = []
        profiles = set(self.plan['profiling_groups'])
        self.assertEqual(len(profiles), 6)
        for owner, groups in ownership(self.plan)['owners'].items():
            with patch('src.stage3.fit_readouts.Recovery') as recovery, \
                 patch('src.stage3.fit_readouts.require_profiles'), \
                 patch('src.stage3.fit_readouts.open_numeric_session') as sessions, \
                 patch('src.stage3.fit_readouts.load_or_publish_statistics', return_value={'reference': {'path': 'stats', 'sha256': 'a'*64}}), \
                 patch('src.stage3.fit_readouts.fit_candidate_block', return_value={'path': 'candidate', 'sha256': 'b'*64}) as fit, \
                 patch('src.stage3.fit_readouts.load_candidate_block', return_value=({'records': [{'status': 'fitted', 'attempts': []}]*25}, None, None, None)), \
                 patch('src.stage3.fit_readouts.select_system', return_value={'path': 'selection', 'sha256': 'c'*64}) as select, \
                 patch('src.stage3.fit_readouts.publish_run_index', return_value={'path': 'index', 'sha256': 'd'*64}):
                instance = recovery.return_value
                instance.statistics, instance.candidates, instance.selections = {}, {}, {}
                instance.entries.return_value = {'statistics': [], 'candidates': [], 'selections': []}
                import numpy as np
                sessions.return_value.read.return_value = ([{}], np.ones((1, 2)))
                self.execute(groups, 'e03' if owner == 'e03' else 'vlm', None if owner == 'e03' else owner, mode='ordinary')
                total.extend((call.args[1], call.args[2], call.args[3]) for call in fit.call_args_list)
                self.assertEqual(select.call_count, len(groups))
        self.assertEqual(len(total), 580)
        self.assertEqual(set(total), {(g, 'true', c) for g in self.plan['groups'] for c in CS})

    def test_measurement_stream_and_notebook_cells(self):
        import hashlib
        events = []
        payload = self.plan.root / 'timing-fixture'
        payload.write_bytes(b'measurement')
        result = measured('fixture', lambda: hashlib.sha256(payload.read_bytes()).hexdigest(),
                          lambda event, **fields: events.append((event, fields)))
        self.assertEqual(result, hashlib.sha256(b'measurement').hexdigest())
        self.assertEqual([e[0] for e in events], ['fixture_start', 'fixture_end'])
        self.assertEqual(set(events[-1][1]['io_hash_exclusive_seconds']), {'read', 'hash', 'write'})
        self.assertGreater(events[-1][1]['io_hash_exclusive_seconds']['read'], 0)
        self.assertGreater(events[-1][1]['io_hash_exclusive_seconds']['hash'], 0)
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'scripts/stage3/fit_readouts.ipynb').read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'cell' + str(index), 'exec')
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__':
    unittest.main()
