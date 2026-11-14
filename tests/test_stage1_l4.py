"""Hardware/configuration guards using software fixtures, not a real L4."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.common.contracts import ContractError

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

if __name__=='__main__':unittest.main()
