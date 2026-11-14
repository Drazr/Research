"""Hardware/configuration guards using software fixtures, not a real L4."""
import importlib.util
from importlib.metadata import PackageNotFoundError
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.common.contracts import ContractError
from src.common.contracts import fingerprint
from src.common.storage import publish, verify
from src.models.hf import validate_device

spec=importlib.util.spec_from_file_location('configure_l4',Path(__file__).resolve().parents[1]/'scripts/stage1/configure_l4.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class L4Tests(unittest.TestCase):
    def torch_fixture(self,name='NVIDIA L4',available=True,memory=24*1024**3,bf16=True):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:available,
            get_device_name=lambda n:name,get_device_properties=lambda n:SimpleNamespace(total_memory=memory),
            is_bf16_supported=lambda:bf16),version=SimpleNamespace(cuda='fixture-only'))

    def test_l4_guard_rejects_cpu_other_gpu_partition_and_precision(self):
        self.assertEqual(module.l4_hardware(self.torch_fixture())['device_name'],'NVIDIA L4')
        for kwargs in [{'available':False},{'name':'other GPU'},{'memory':8*1024**3},{'bf16':False}]:
            with self.assertRaises(ContractError):module.l4_hardware(self.torch_fixture(**kwargs))

    def test_exact_versions_record_build_suffix_and_reject_drift(self):
        expected={name:'1.0.0' for name in module.PACKAGES}
        with patch.object(module,'version',return_value='1.0.0+fixture'):
            actual=module.check_versions({'software_versions':expected})
            self.assertEqual(actual['torch'],'1.0.0+fixture')
        with patch.object(module,'version',return_value='2.0.0'):
            with self.assertRaises(ContractError):module.check_versions({'software_versions':expected})

    def test_adapter_gpu_preflight_before_any_download(self):
        cfg={'device':'cuda:0','precision':'bfloat16'}
        validate_device(self.torch_fixture(),cfg)
        for kwargs in [{'available':False},{'bf16':False}]:
            with self.assertRaises(ContractError):validate_device(self.torch_fixture(**kwargs),cfg)

    def test_blocked_configuration_report_persists_with_compact_windows_safe_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'a_reasonably_long_local_development_root'
            protocol=json.loads((module.REPO/'configs/protocol.json').read_text())
            meta=dict(kind='stage1_dataset',status='passed',run_id='fixture',code_commit='0'*40,
                protocol_fingerprint=fingerprint(protocol),dataset_fingerprint='b'*64,
                model_fingerprint=None,seeds={},software_versions={'fixture':'only'},upstream_artifacts=[])
            dataset=publish(root,'shared/fixture',meta,{'fixture.json':b'{}'})
            argv=['configure_l4.py','--model','qwen','--phase1-root',str(root),
                  '--dataset-manifest',str(dataset/'_COMPLETE.json'),'--code-commit','0'*40,
                  '--storage-scope','local-development']
            with patch.object(module.sys,'argv',argv), \
                    patch.object(module,'check_versions',side_effect=PackageNotFoundError('fixture-missing-runtime')), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(),2)
            markers=list((root/'cache/runtime-configurations/qwen').glob('*/_COMPLETE.json'))
            self.assertEqual(len(markers),1)
            self.assertEqual(verify(markers[0].parent,required_status=None)['status'],'blocked')
            report_path=markers[0].parent/'configuration_report.json'
            report=json.loads(report_path.read_text())
            self.assertFalse(report['weights_loaded']);self.assertTrue(report['blockers'])
            self.assertLess(len(str(report_path)),260)

if __name__=='__main__':unittest.main()
