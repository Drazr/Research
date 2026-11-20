"""Unit and fixture tests for Phase 3 CPU Track Stage A1 foundation."""

import json
from pathlib import Path
import tempfile
import unittest

from src.common.contracts import ContractError, fingerprint
from src.phase3.interfaces import (
    CERTIFIED_PHASE2_FOR_PHASE3_CPU,
    KIND_PROBE,
    KIND_PROBE_PAYLOAD,
    PHASE3_CPU_DECISION_RANGES,
    PHASE3_PARENT_DIRS,
    REPO,
    SCHEMA_PROBE,
    init_phase3_dirs,
    load_phase3_artifact,
    plain,
    publish_phase3,
    require_open_split,
    require_phase3_namespace,
    storage_probe,
    verify_resume_ref,
    verify_storage_probe,
)


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.14", "numpy": "2.1.3"},
    }


class Phase3CpuFoundationTests(unittest.TestCase):
    def test_namespace_and_confirm_guards(self):
        self.assertEqual(require_open_split("SCREEN"), "SCREEN")
        with self.assertRaises(ContractError):
            require_open_split("CONFIRM")
        with self.assertRaises(ContractError):
            require_open_split("UNKNOWN")

        self.assertEqual(require_phase3_namespace("phase3/results/e40/qwen"), "phase3/results/e40/qwen")
        for bad in ("phase2/plans", "results/stage4", "../phase3/x", "/phase3/x", "phase3\\x"):
            with self.assertRaises(ContractError):
                require_phase3_namespace(bad)

    def test_init_dirs_publish_load_and_resume_rejection(self):
        protocol = _load_protocol()
        ds_fp = fingerprint({"test": "phase3"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = init_phase3_dirs(root)
            self.assertEqual(created, PHASE3_PARENT_DIRS)
            for rel in PHASE3_PARENT_DIRS:
                self.assertTrue((root / rel).is_dir())

            ref = publish_phase3(
                root,
                protocol,
                ds_fp,
                "phase3/results/e40/qwen",
                "phase3_analysis_block",
                {"schema": "phase3.analysis.v1", "plan_identity": "abc", "model": "qwen", "inputs": {"x": 1}},
                _fake_producer(),
            )
            _, body, _ = load_phase3_artifact(
                root,
                protocol,
                ref,
                "phase3_analysis_block",
                expected_schema="phase3.analysis.v1",
                expected_dataset=ds_fp,
                expected_plan_identity="abc",
                expected_owner="qwen",
            )
            self.assertEqual(body["model"], "qwen")

            with self.assertRaises(ContractError):
                verify_resume_ref(
                    root, protocol, ref, "phase3_analysis_block", expected_owner="gemma"
                )
            with self.assertRaises(ContractError):
                verify_resume_ref(
                    root, protocol, ref, "phase3_analysis_block", expected_inputs={"x": 2}
                )

    def test_two_process_storage_probe(self):
        protocol = _load_protocol()
        ds_fp = fingerprint({"probe": "phase3"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_phase3_dirs(root)
            probe_ref = storage_probe(root, protocol, ds_fp, _fake_producer())
            _, body = verify_storage_probe(root, protocol, probe_ref, ds_fp)
            self.assertTrue(body["independent_process_readback"])
            self.assertNotEqual(body["reader_pid"], body["writer_pid"])
            load_phase3_artifact(
                root, protocol, probe_ref, KIND_PROBE, expected_schema=SCHEMA_PROBE, expected_dataset=ds_fp
            )
            load_phase3_artifact(
                root, protocol, body["payload"], KIND_PROBE_PAYLOAD, expected_schema=SCHEMA_PROBE, expected_dataset=ds_fp
            )

    def test_certified_refs_and_decision_ranges_complete(self):
        self.assertIn("phase2_plan", CERTIFIED_PHASE2_FOR_PHASE3_CPU)
        self.assertIn("phase2_e17_auth", CERTIFIED_PHASE2_FOR_PHASE3_CPU)
        for m in ("qwen", "gemma"):
            self.assertEqual(
                set(CERTIFIED_PHASE2_FOR_PHASE3_CPU["models"][m].keys()),
                {"e29_analysis", "e29_gate", "e30_analysis", "e11_analysis", "e17_analysis"},
            )
        self.assertEqual(set(PHASE3_CPU_DECISION_RANGES.keys()), {"E40", "E41", "E44", "E12"})

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/prepare_phase3_cpu.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 4)
        full = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("CODE_REF = 'phase3'", full)
        self.assertIn("src.phase3.cpu_plan", full)


if __name__ == "__main__":
    unittest.main()
