"""CPU routing/restart fixtures. These never certify real L4 inference."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from test_stage2_foundation import ArtifactFixture, BridgeFixture, source_row
from src.common.contracts import ContractError, LANGUAGES
from src.stage2.schema import CONDITIONS, SPLITS, plain, reference
from src.stage2.interfaces import load_feature_release, verify_feature_chunk
from src.stage2.extract_features import Extractor, bind_inputs, load_resume, safe_error, sites_for


class WorkerTests(ArtifactFixture):
    def worker(self, indexes=(), emit=None):
        # Real serializers/checksums/audits; only upstream acceptance/runtime are fixtures.
        inputs = bind_inputs(self.plan, self.inputs + [source_row(99, 'en', 'CONFIRM')])
        return Extractor(self.root, self.protocol, self.plan, inputs, self.producer('worker'), indexes, emit or (lambda _: None))

    def setUp(self):
        super().setUp()
        from test_stage2_foundation import PROTOCOL
        self.protocol = PROTOCOL
        self.runtime_patch = patch('src.stage2.extract_features.assert_runtime')
        self.runtime = self.runtime_patch.start()

    def tearDown(self):
        self.runtime_patch.stop()
        super().tearDown()

    def test_three_conditions_shared_forward_and_full_resumed_release(self):
        adapter = BridgeFixture()
        first = self.worker().run(lambda: adapter)
        self.assertEqual(first['status'], 'passed')
        self.assertFalse(first['stage2_complete'])
        self.assertEqual(len(first['feature_releases']), 12)
        # 36 core forwards + nine initial public checks + 20 boundary repeats.
        self.assertEqual(len(adapter.calls), 65)
        self.assertTrue(all(not c[2] and c[3] == 'preliminary_features' for c in adapter.calls))
        self.assertEqual({c[1] for c in adapter.calls}, set(CONDITIONS))
        for ref in first['feature_releases']:
            path = self.root/ref['path']
            body = json.loads((path.parent/'release.json').read_text())
            role = dict(zip(SPLITS, ('source_fit','source_select','calibration','exploratory_score')))[body['split']]
            release = load_feature_release(path, self.root, self.protocol, role,
                {'plan_fingerprint':self.plan['plan_fingerprint'], 'condition':body['condition'], 'split':body['split']})
            key = next(iter(release.site_catalog))
            rows, matrix = release.read_matrix(key)
            self.assertEqual(matrix.dtype.str, '<f4')
            self.assertEqual(len(rows), 1 if body['condition'] == 'image_only' else 4)
        def no_load():
            self.fail('Verified full resume must not load a model')
        resumed = self.worker([first['resume_index']]).run(no_load)
        self.assertEqual(len(resumed['feature_releases']), 12)
        self.assertEqual(len(adapter.calls), 65)

    def test_crash_after_image_publication_recovers_first_measurements(self):
        events = []
        def crash(event):
            events.append(event)
            if event.get('event') == 'resume_index' and len([e for e in events if e.get('event') == 'resume_index']) == 2:
                raise RuntimeError('simulated crash after image-only publication')
        adapter = BridgeFixture()
        worker = self.worker(emit=crash)
        with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
            worker.run(lambda: adapter)
        index = worker.latest_progress
        accepted, issues, resource, measurements = load_resume([index], self.root, self.protocol, self.plan)
        self.assertEqual(set(accepted), {('FIT_0000', 'image_only')})
        self.assertIn('image_only', measurements)
        self.assertIsNone(resource)
        restarted_adapter = BridgeFixture()
        result = self.worker([index]).run(lambda: restarted_adapter)
        self.assertEqual(result['status'], 'passed')
        # First image-only reference/public/repeat are already accepted.
        self.assertEqual(len(restarted_adapter.calls), 62)

    def test_corrupt_child_is_reextracted_without_mutating_other_chunks(self):
        first = self.worker().run(BridgeFixture)
        body = json.loads((Path(first['resume_index']).parent/'progress.json').read_text())
        target = next(r for r in body['chunks'] if 'DEV-S/caption_only' in r['path'])
        marker = self.root/target['path']
        (marker.parent/'site_000.npy').write_bytes(b'corrupt')
        adapter = BridgeFixture()
        worker = self.worker([first['resume_index']])
        self.assertEqual(len(worker.issues), 1)
        result = worker.run(lambda: adapter)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(adapter.calls), 6)  # four core plus first/last
        self.assertEqual((marker.parent/'site_000.npy').read_bytes(), b'corrupt')
        self.assertNotEqual(worker.accepted[('DEV-S_0000','caption_only')][0], target)

    def test_visual_failure_stops_before_publication_and_records_context(self):
        class ChangingVisual(BridgeFixture):
            def _forward(self, row, sites, condition='image_caption', **kwargs):
                values, item = super()._forward(row, sites, condition, **kwargs)
                if condition == 'image_caption':
                    values['encoder:None:visual_mean'] = [99., 99.]
                return values, item
        worker = self.worker()
        with self.assertRaisesRegex(ContractError, 'Visual context'):
            worker.run(ChangingVisual)
        self.assertFalse(worker.accepted)
        diagnostics = list(self.root.glob('results/stage2-validation/*/report.json'))
        body = json.loads(diagnostics[-1].read_text())
        self.assertEqual(body['context']['image_id'], source_row().image_id)
        self.assertFalse(body['context']['visual_comparison']['equal'])

    def test_runtime_mismatch_precedes_all_forwards(self):
        self.runtime.side_effect = ContractError('actual runtime differs')
        adapter = BridgeFixture()
        with self.assertRaisesRegex(ContractError, 'runtime differs'):
            self.worker().run(lambda: adapter)
        self.assertFalse(adapter.calls)

    def test_input_joins_refuse_caption_changes_missing_languages_and_confirm(self):
        changed = [replace(r,caption_text='substitute') if r.language == 'bn' else r for r in self.inputs]
        for records in (changed, self.inputs[1:], [replace(self.inputs[0], split='CONFIRM')]+self.inputs[1:]):
            with self.assertRaises(ContractError):
                bind_inputs(self.plan, records)
        self.assertEqual(len(bind_inputs(self.plan, self.inputs+[source_row(99, 'en', 'CONFIRM')])), 16)
        self.assertEqual(len(sites_for(self.plan, 'image_caption')), 3)
        self.assertEqual(len(sites_for(self.plan, 'caption_only')), 2)

    def test_failure_preserves_prior_release_and_secret_redaction(self):
        class Overlong(BridgeFixture):
            def _forward(self, row, sites, condition='image_caption', **kwargs):
                if row.split == 'DEV-S':
                    raise ContractError('over-limit input hf_FAKESECRET')
                return super()._forward(row, sites, condition, **kwargs)
        worker = self.worker()
        with self.assertRaisesRegex(ContractError, 'over-limit'):
            worker.run(Overlong)
        self.assertEqual(len(worker.releases), 3)
        self.assertEqual(len(worker.accepted), 3)
        for p in self.root.glob('results/stage2-validation/*/report.json'):
            self.assertNotIn('hf_FAKESECRET', p.read_text())
        with patch.dict('os.environ', {'HF_TOKEN':'arbitrary-secret'}):
            self.assertEqual(safe_error(RuntimeError('arbitrary-secret')), '[REDACTED]')

    def test_32_image_storage_boundary_and_unpadded_tail_are_batch_one(self):
        from src.stage2.plans import compile_inventory
        from src.stage2.schema import frozen, publish_json, metadata
        self.inputs = [source_row(i, l) for i in range(1,34) for l in LANGUAGES]
        plan = plain(self.plan)
        plan.update(compile_inventory(SimpleNamespace(records=self.inputs)))
        plan.pop('manifest_reference')
        marker = publish_json(self.root, 'plans/boundary', metadata('stage2_model_plan', self.protocol,
                              plan['dataset_fingerprint'], self.producer(), 'f'*64), 'plan.json', plan)
        plan['manifest_reference'] = reference(marker, self.root)
        self.plan = frozen(plan)
        with patch('src.stage2.features.load_job_plan', return_value=self.plan):
            worker = self.worker()
            adapter = BridgeFixture()
            from src.stage2.interfaces import extract_with_evidence, compare_public_path
            def one(function):
                def call(adapter, request):
                    self.assertEqual(len(request.inputs), 1)
                    return function(adapter, request)
                return call
            with patch('src.stage2.extract_features.extract_with_evidence', side_effect=one(extract_with_evidence)), \
                 patch('src.stage2.extract_features.compare_public_path', side_effect=one(compare_public_path)):
                worker.run(lambda: adapter)
            self.assertEqual(len(adapter.calls), 317)
            for (chunk, condition), refs in worker.accepted.items():
                _, rows, _, _ = verify_feature_chunk(self.root/refs[0]['path'], self.root, self.protocol, self.plan)
                images = 32 if chunk == 'FIT_0000' else 1
                self.assertEqual(len(rows), images * (1 if condition == 'image_only' else 4))
                self.assertEqual([r['language'] for r in rows[:4]],
                                 [None]*min(4,len(rows)) if condition == 'image_only' else list(LANGUAGES))

    def test_conflicting_resume_vectors_stop_before_model_load(self):
        from src.stage2.interfaces import publish_feature_chunk
        refs = []
        for i in range(2):
            chunk, files = self.payload(delta=i)
            marker = publish_feature_chunk(self.root, self.plan, chunk, 'caption_only', 'conflict'+str(i), files, self.producer())
            refs.append(reference(marker,self.root))
        worker = self.worker()
        worker.accepted[('FIT_0000','caption_only')] = refs
        worker.snapshot()
        with self.assertRaisesRegex(ContractError, 'reproducibility conflict'):
            self.worker([worker.latest_progress])

    def test_resume_between_first_chunks_and_resource_report(self):
        worker = self.worker()
        with patch('src.stage2.extract_features.publish_resource_report', side_effect=RuntimeError('resource crash')):
            with self.assertRaisesRegex(RuntimeError, 'resource crash'):
                worker.run(BridgeFixture)
        self.assertEqual(len(worker.accepted), 3)
        adapter = BridgeFixture()
        result = self.worker([worker.latest_progress]).run(lambda: adapter)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(len(adapter.calls), 42)

    def test_boundary_repeat_failure_blocks_affected_chunk(self):
        class Drifting(BridgeFixture):
            def _forward(self, row, sites, condition='image_caption', **kwargs):
                seen = sum(k == row.row_key and c == condition for k,c,_,_ in self.calls)
                values, item = super()._forward(row, sites, condition, **kwargs)
                if condition == 'image_only' and seen >= 2:
                    values = {k:[2.,3.] for k in values}
                return values,item
        worker = self.worker()
        with self.assertRaisesRegex(ContractError,'Boundary repeat'):
            worker.run(Drifting)
        self.assertFalse(worker.accepted)
        diagnostic = next(self.root.glob('results/stage2-validation/*/report.json'))
        comparison = json.loads(diagnostic.read_text())['context']['repeat_comparison']
        self.assertNotEqual(comparison['expected_digest'], comparison['observed_digest'])
        self.assertEqual(comparison['observed_vector'], [2.,3.])

    def test_notebook_cells_compile_and_have_no_saved_outputs(self):
        from src.stage2.schema import REPO
        notebook = json.loads((REPO/'scripts/stage2/extract_features.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()
