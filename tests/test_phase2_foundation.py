"""Unit and fixture checks for Phase 2 Stage 1 shared foundation."""

import json
from pathlib import Path
import tempfile
import unittest

from src.common.contracts import ContractError, fingerprint
from src.phase2.interfaces import (
    ALLOWED_SPLITS,
    KIND_E07_AUDIT,
    KIND_PLAN,
    KIND_PROBE,
    MODELS,
    PHASE2_PARENT_DIRS,
    SCHEMA_E07_AUDIT,
    SCHEMA_PLAN,
    init_phase2_dirs,
    load_phase2_artifact,
    publish_phase2,
    require_open_split,
    require_phase2_namespace,
    storage_probe,
    verify_resume_ref,
    verify_storage_probe,
)
from src.stage2.schema import REPO, plain

try:
    import numpy as np
except ImportError:
    np = None


class SchemaAndStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        self.dataset_fp = "a" * 64
        self.producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0"},
        }

    def test_split_and_namespace_guards(self):
        for split in ALLOWED_SPLITS:
            self.assertEqual(require_open_split(split), split)
        with self.assertRaisesRegex(ContractError, "CONFIRM is closed"):
            require_open_split("CONFIRM")
        with self.assertRaisesRegex(ContractError, "Unknown or unauthorized"):
            require_open_split("TEST")

        self.assertEqual(require_phase2_namespace("phase2/plans"), "phase2/plans")
        for bad in ("results/stage4", "predictions/stage4", "phase2/../results", "/phase2/plans", "phase2\\plans", ""):
            with self.assertRaises(ContractError):
                require_phase2_namespace(bad)

    def test_init_phase2_dirs_and_publish_load_resume(self):
        created = init_phase2_dirs(self.root)
        self.assertEqual(created, PHASE2_PARENT_DIRS)
        for rel in PHASE2_PARENT_DIRS:
            self.assertTrue((self.root / rel).is_dir())
        # Phase 1 directories are not created by Phase 2 initialization.
        self.assertFalse((self.root / "results").exists())

        body = {
            "schema": SCHEMA_E07_AUDIT,
            "plan_identity": "b" * 64,
            "owner": "qwen",
            "inputs": {"dataset": self.dataset_fp},
            "verdict": "passed_full",
        }
        ref = publish_phase2(
            self.root,
            self.protocol,
            self.dataset_fp,
            "phase2/audits/e07",
            KIND_E07_AUDIT,
            body,
            self.producer,
            upstream=[{"path": "results/stage4/releases/x/_COMPLETE.json", "sha256": "c" * 64}],
        )
        loaded_ref, loaded_body, marker = load_phase2_artifact(
            self.root,
            self.protocol,
            ref,
            KIND_E07_AUDIT,
            expected_schema=SCHEMA_E07_AUDIT,
            expected_dataset=self.dataset_fp,
            expected_plan_identity="b" * 64,
            expected_owner="qwen",
        )
        self.assertEqual(plain(loaded_ref), plain(ref))
        self.assertEqual(loaded_body["verdict"], "passed_full")
        self.assertEqual(marker["kind"], KIND_E07_AUDIT)

        # Resume accepts exact match and rejects foreign owner, wrong plan, wrong inputs, or tampering.
        verify_resume_ref(
            self.root,
            self.protocol,
            ref,
            KIND_E07_AUDIT,
            expected_schema=SCHEMA_E07_AUDIT,
            expected_dataset=self.dataset_fp,
            expected_plan_identity="b" * 64,
            expected_owner="qwen",
            expected_inputs={"dataset": self.dataset_fp},
        )
        with self.assertRaisesRegex(ContractError, "Foreign Phase 2 owner"):
            verify_resume_ref(self.root, self.protocol, ref, KIND_E07_AUDIT, expected_owner="gemma")
        with self.assertRaisesRegex(ContractError, "plan_identity mismatch"):
            verify_resume_ref(self.root, self.protocol, ref, KIND_E07_AUDIT, expected_plan_identity="f" * 64)
        with self.assertRaisesRegex(ContractError, "inputs do not match"):
            verify_resume_ref(self.root, self.protocol, ref, KIND_E07_AUDIT, expected_inputs={"dataset": "d" * 64})
        with self.assertRaisesRegex(ContractError, "schema mismatch"):
            load_phase2_artifact(self.root, self.protocol, ref, KIND_E07_AUDIT, expected_schema=SCHEMA_PLAN)
        with self.assertRaisesRegex(ContractError, "manifest checksum mismatch"):
            load_phase2_artifact(self.root, self.protocol, {"path": ref["path"], "sha256": "0" * 64}, KIND_E07_AUDIT)

        # Tampering with body.json fails checksum verification.
        body_path = (self.root / ref["path"]).parent / "body.json"
        body_path.write_text('{"tampered": true}\n', encoding="utf-8")
        with self.assertRaises(ContractError):
            load_phase2_artifact(self.root, self.protocol, ref, KIND_E07_AUDIT)

    def test_two_process_storage_probe(self):
        probe_ref = storage_probe(self.root, self.protocol, self.dataset_fp, self.producer)
        verified_ref, body = verify_storage_probe(self.root, self.protocol, probe_ref, self.dataset_fp)
        self.assertEqual(plain(verified_ref), plain(probe_ref))
        self.assertTrue(body["independent_process_readback"])
        self.assertNotEqual(body["reader_pid"], body["writer_pid"])
        with self.assertRaises(ContractError):
            verify_storage_probe(self.root, self.protocol, probe_ref, "f" * 64)
        with self.assertRaises(ContractError):
            load_phase2_artifact(self.root, self.protocol, probe_ref, KIND_PLAN)
        self.assertEqual(load_phase2_artifact(self.root, self.protocol, probe_ref, KIND_PROBE)[0], probe_ref)


@unittest.skipIf(np is None, "numpy/scipy/sklearn required for full Phase 1 fixture tests")
class Phase1BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        tests_dir = str(REPO / "tests")
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        import test_stage5_foundation as stage5
        from src.common.contracts import LANGUAGES
        from src.common.storage import file_sha256
        from src.phase2.interfaces import bind_phase1

        stage5.ReportingTests.setUpClass()
        cls.addClassCleanup(stage5.ReportingTests.doClassCleanups)
        cls.plan = stage5.ReportingTests.plan
        cls.release = stage5.ReportingTests.release
        cls.root = cls.plan.root
        cls.protocol = cls.plan.protocol
        cls.binding = bind_phase1(cls.root, cls.protocol, cls.release)

    def test_binding_and_resolvers(self):
        from src.phase2.interfaces import (
            bind_phase1,
            e17_inputs,
            e29_screen_blocks,
            load_frozen_source_head,
            midpoint_depth,
            read_verified_midpoint_features,
        )

        b = self.binding
        self.assertGreater(b.g0_p_value, 0.01)
        self.assertEqual(len(b.lineage_identity), 64)
        self.assertEqual(len(b.bootstrap_sha256), 64)
        for model in MODELS:
            self.assertEqual(midpoint_depth(b, model), 2)
            blocks = e29_screen_blocks(b, model)
            count = sum(len(blocks[c][p]) for c in blocks for p in blocks[c])
            self.assertEqual(count, 16)

        with self.assertRaisesRegex(ContractError, "CONFIRM is closed"):
            read_verified_midpoint_features(b, "qwen", "image_caption", "last_caption", "en", "CONFIRM")
        with self.assertRaises(ContractError):
            bind_phase1(self.root, self.protocol, {"path": self.release["path"], "sha256": "0" * 64})

        # Qwen midpoint last_caption is scored in the Stage 4 fixture; verify e17_inputs + frozen head load cleanly.
        e17 = e17_inputs(b, "qwen")
        self.assertEqual(len(e17["screen_blocks"]), 4)
        self.assertEqual(len(e17["dev_c_diagonals"]), 4)
        self.assertEqual(len(e17["dev_c_feature_releases"]), 4)
        self.assertEqual(len(e17["source_systems"]), 4)
        self.assertEqual(sum(len(v) for v in e17["off_diagonal_targets"].values()), 12)
        coef, intercept, estimable, prevalence, stats = load_frozen_source_head(
            b, "qwen", "image_caption", "last_caption", "en"
        )
        self.assertEqual(coef.shape[0], 25)
        self.assertEqual(intercept.shape, (25,))
        self.assertEqual(estimable.shape, (25,))
        self.assertEqual(prevalence.shape, (25,))
        self.assertIn("scale", stats)


if __name__ == "__main__":
    unittest.main()
