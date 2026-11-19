"""Stage 4 integration on the complete synthetic route inventory."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.stage2.schema import plain
from src.stage4.interfaces import load_stage4_release, prepare_stage4, validate_stage4
from src.stage4.run_e01_e04 import run_e01_e04
from src.stage4.run_e19 import run_e19
from src.stage4.run_e25 import run_e25
from src.stage4.run_scoring import run_scoring
from test_stage4_scoring import FullStage4Fixture


class ReleaseIntegrationTests(unittest.TestCase):
    def test_worker_outputs_form_one_complete_release(self):
        depths = {"qwen": (1, 2, 3, 4), "gemma": (1, 2, 3, 4)}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("src.stage4.plan.DEPTHS", depths),
            patch("src.stage4.plan.BOOTSTRAP_DRAWS", 20),
            patch("src.stage4.plan.PERMUTATION_DRAWS", 16),
            patch("src.stage4.metrics.PERMUTATION_DRAWS", 16),
            patch("src.stage4.run_e25.PERMUTATION_DRAWS", 16),
        ):
            fixture = FullStage4Fixture(directory)
            f = fixture.fixture
            plan = prepare_stage4(f.root, f.protocol, fixture.registry, f.producer)
            predictions = []
            handoffs = {}

            def report(name, **data):
                if name == "priority_handoff":
                    handoffs[data["owner"]] = data

            for model in ("qwen", "gemma"):
                summary = run_scoring(plan, "model", f.producer, model=model, report=report)
                predictions.extend(summary["predictions"])
            summary = run_scoring(plan, "e03", f.producer, report=report)
            predictions.extend(summary["predictions"])
            self.assertEqual(len(predictions), 153)

            shared = run_e01_e04(plan, predictions, f.producer, owner="e03")
            analyses = list(shared["analyses"])
            for model in ("qwen", "gemma"):
                priority = handoffs[model]
                e19 = run_e19(plan, model, priority["predictions"], f.producer, progress=lambda *a, **kw: None)
                e25 = run_e25(plan, model, priority["midpoint_screen_refs"], f.producer)
                core = run_e01_e04(plan, predictions, f.producer, owner=model)
                analyses.extend(e19["units"].values())
                analyses.extend(e25["analyses"])
                analyses.extend(core["analyses"])
            self.assertEqual(len(analyses), 47)

            protocol_file = Path(directory) / "fixture-protocol.json"
            protocol_file.write_text(json.dumps(f.protocol), encoding="utf-8")
            run = subprocess.run

            def child_readback(command, **kwargs):
                # Use the real independent reader in a fresh process with the fixture's protocol.
                self.assertEqual(command[2:5], ["-m", "src.stage4.operations", "readback"])
                code = (
                    "import json,sys; from pathlib import Path; "
                    "from src.stage4.validation import independent_readback; "
                    "independent_readback(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), "
                    "json.loads(sys.argv[3]))"
                )
                return run([command[0], "-c", code, str(f.root), str(protocol_file), command[-1]], **kwargs)

            with patch("src.stage4.validation.subprocess.run", side_effect=child_readback):
                ref = validate_stage4(plan, predictions, analyses, f.producer)
            release = load_stage4_release(plan, ref)
            self.assertTrue(release["stage4_complete"])
            self.assertEqual(release["route_counts"], {"SCREEN": 529, "DEV-C": 8})
            self.assertEqual(release["analysis_units"], 47)
            self.assertEqual(release["stage3_registry"], plain(plan["registry"]))
