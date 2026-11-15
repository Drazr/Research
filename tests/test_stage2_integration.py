"""Suite-index orchestration fixtures; not production/model acceptance."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.stage2.operations import validate_suite
from src.stage2.schema import (CONDITIONS, SPLITS, metadata, publish_json, reference)
from test_stage2_foundation import producer, PROTOCOL


class SuiteIndexTests(unittest.TestCase):
    def test_complete_inventory_separates_resource_receipts_and_missing_inputs_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def artifact(kind, body, filename):
                run = 'fixture_' + str(len(list(root.iterdir())))
                return publish_json(root, run, metadata(kind, PROTOCOL, 'b'*64,
                                    producer(run)), filename, body)
            data_marker = artifact('stage2_data_plan', {}, 'plan.json')
            data = {'kind': 'stage2_data_plan', 'dataset_fingerprint': 'b'*64,
                    'plan_fingerprint': 'd'*64, 'manifest_reference': reference(data_marker, root),
                    'recipes': [{'recipe_fingerprint': str(i)} for i in range(13)]}
            plans = {str(data_marker): data}
            inventory = {'data_plan': data_marker, 'model_plans': {},
                         'feature_releases': {}, 'resource_reports': {}, 'text_families': []}
            for model in ('qwen', 'gemma'):
                marker = artifact('stage2_model_plan', {}, 'plan.json')
                plans[str(marker)] = {'model': model, 'dataset_fingerprint': 'b'*64,
                    'plan_fingerprint': model, 'references': {'data_plan': reference(data_marker, root)}}
                inventory['model_plans'][model] = marker
                inventory['resource_reports'][model] = artifact('stage2_resource_report', {}, 'resources.json')
                inventory['feature_releases'][model] = [artifact('stage2_feature_release',
                    {'condition': c, 'split': s}, 'release.json') for c in CONDITIONS for s in SPLITS]
            for i in range(13):
                inventory['text_families'].append(artifact('stage2_text_family',
                    {'recipe': {'recipe_fingerprint': str(i)}, 'preprocessing_fingerprint': 'c'*64,
                     'splits': dict.fromkeys(SPLITS)}, 'family.json'))
            # Payload correctness is exercised by the shared/worker suites. Here
            # isolate final orchestration and its inventory separation/coverage.
            def features(marker, root, protocol, role, expected):
                return SimpleNamespace(condition=expected['condition'], split=expected['split'])
            with patch('src.stage2.plans.load_job_plan', side_effect=lambda p, *_: plans[str(p)]), \
                 patch('src.stage2.features.load_feature_release', side_effect=features), \
                 patch('src.stage2.text.load_text_family', return_value=SimpleNamespace(read_matrix=lambda _: None)), \
                 patch('src.stage2.operations.verify_resource_report'):
                result = validate_suite(root, PROTOCOL, inventory, producer('accepted'))
                body = json.loads((result.parent/'report.json').read_text())
                self.assertEqual(body['status'], 'passed')
                self.assertEqual(len(body['feature_releases']), 24)
                self.assertEqual(set(body['resource_reports']), {'qwen', 'gemma'})
                self.assertTrue(all(json.loads((root/r['path']).read_text())['kind'] == 'stage2_feature_release'
                                    for r in body['feature_releases']))
                meta = json.loads(result.read_text())
                self.assertTrue(all(r in meta['upstream_artifacts'] for r in body['resource_reports'].values()))
                inventory['feature_releases']['qwen'].pop()
                result = validate_suite(root, PROTOCOL, inventory, producer('missing_release'))
                self.assertEqual(json.loads(result.read_text())['status'], 'blocked')
                inventory['resource_reports'].pop('gemma')
                result = validate_suite(root, PROTOCOL, inventory, producer('missing_resource'))
                self.assertIn('gemma: measured first-chunk resource report missing',
                              json.loads((result.parent/'report.json').read_text())['blockers'])
