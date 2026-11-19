"""Fixture tests for the Stage 5 per-model reporting worker and notebook."""

import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError, LANGUAGES
from src.stage2.schema import REPO, plain
from src.stage4.analysis import e25_memberships
from src.stage4.interfaces import DEPTHS
from src.stage4.validation import OPTIONAL_COVERAGE, PENDING, RELEASE, _publish
from src.stage5.coordinator import SYNTHESIS_KIND, publish_synthesis
from src.stage5.interfaces import OWNERS, REPORT_KIND, publish_owner_report
from src.stage5.run_model_report import (
    EXPECTED_UNIT_COUNTS,
    FAMILIES,
    MODELS,
    assemble_family_views,
    expected_model_units,
    format_model_report_summary,
    main as cli_main,
    run_model_report,
    summarize_model_report,
    validate_model_report_body,
)

import test_stage4_foundation as stage4


def _fixture_analysis_rows(unit):
    """Build deterministic fixture rows preserving intervals, support, and status for every comparison."""
    rows = []
    for cid, comp in unit["comparisons"].items():
        metric = comp["statistics"][0]
        row = {
            "comparison_id": cid,
            "category": "primary_macro",
            "statistic": metric,
            "value": 0.5,
            "status": "ok",
            "reason": None,
            "support": {"known": 4, "categories": 14},
            "interval": {"lower": 0.4, "upper": 0.6, "valid_draws": 2000, "total_draws": 2000},
        }
        rows.append(row)
    return rows


def _fixture_e19_cutoffs(plan):
    records = []
    for baseline in ("probe", "fit_prior", "always_absent"):
        for category in plan["categories"]:
            for nominal in (0.5, 0.75, 0.9):
                records.append({
                    "baseline": baseline,
                    "category": category,
                    "nominal": nominal,
                    "eligible": 20,
                    "retained": 18,
                    "cutoff": 0.75,
                })
    return records


class ModelReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = stage4.AcceptedInputTests
        fixture.setUpClass()
        cls.addClassCleanup(fixture.doClassCleanups)
        cls.plan = plan = fixture.plan
        cls.producer = dict(fixture.fixture.producer, storage_scope="local-development")
        cls.units = units = stage4.analysis_units(plan)

        def stub(prefix, key):
            return {"path": f"fixture/{prefix}/{key}/_COMPLETE.json", "sha256": "0" * 64}

        body = {
            "schema": "stage4.release.v1",
            "plan": plain(plan.reference),
            "plan_identity": plan["identity"],
            "dataset_fingerprint": plan["dataset_fingerprint"],
            "stage3_registry": plain(plan["registry"]),
            "g0_gate": plain(plan["g0_gate"]),
            "stage4_complete": False,
            "optional_coverage": copy.deepcopy(OPTIONAL_COVERAGE),
            "analyses": {uid: stub("analysis", uid) for uid in units},
            "predictions": {bid: stub("prediction", bid) for bid in plan["blocks"]},
        }

        def write_release(kind, payload):
            return _publish(plan.store, plan["dataset_fingerprint"], "fixture/stage5-model", kind,
                            payload, {}, [plan.reference], cls.producer)

        cls.pending_release = write_release(PENDING, body)
        cls.release = write_release(RELEASE, dict(body, stage4_complete=True,
                                                  independent_readback=cls.pending_release))

        memberships = e25_memberships(plan)
        cutoffs = _fixture_e19_cutoffs(plan)
        cls.fixture_analysis = {}
        for uid, unit in units.items():
            cls.fixture_analysis[body["analyses"][uid]["path"]] = {
                "unit_id": uid,
                "predictions": {bid: body["predictions"][bid] for bid in unit["required_blocks"]},
                "rows": _fixture_analysis_rows(unit),
                "cutoffs": cutoffs if unit["experiment"] == "E19" else [],
                "association_memberships": memberships if unit["experiment"] == "E25" else {},
            }

        def read_artifact(plan_arg, reference, *, units):
            return cls.fixture_analysis[reference["path"]]

        patcher = patch("src.stage5.interfaces.load_analysis_block", side_effect=read_artifact)
        patcher.start()
        cls.addClassCleanup(patcher.stop)

    def test_independent_qwen_and_gemma_completeness_and_six_families(self):
        published = {}
        for model in MODELS:
            events = []
            result = run_model_report(
                self.plan.root, self.plan.protocol, self.release, model, self.producer,
                progress=events.append,
            )
            self.assertFalse(result["resumed"])
            ref = result["reference"]
            self.assertTrue(ref["path"].startswith(f"results/stage5/reports/{model}/"))
            self.assertEqual(len(events), 23)
            self.assertTrue(all(e["event"] == "report_unit_read" and e["owner"] == model for e in events))

            body = self.plan.store.json(ref, "body.json", REPORT_KIND)
            self.assertEqual(body["schema"], "stage5.owner_report.v1")
            self.assertEqual(body["owner"], model)
            self.assertEqual(set(body["tables"]), set(expected_model_units(model)))
            self.assertEqual(len(body["tables"]), 23)

            views = assemble_family_views(body)
            self.assertEqual(views["depths"], list(DEPTHS[model]))
            self.assertEqual(set(views["families"]), set(FAMILIES))
            for family, expected_count in EXPECTED_UNIT_COUNTS.items():
                self.assertEqual(views["families"][family]["unit_count"], expected_count)

            # E01: all 4 depths, 16 routes (absolute + fit_prior + always_absent) + 12 transfer gaps = 60/depth.
            e01_kinds = views["families"]["E01"]["comparison_kinds"]
            self.assertEqual(e01_kinds, {
                "absolute": 64, "always_absent": 64, "fit_prior": 64, "transfer_gap": 48,
            })
            # E02: 4 depths + visual unit (encoder/projector image_only + visual_delta).
            e02_kinds = views["families"]["E02"]["comparison_kinds"]
            self.assertEqual(e02_kinds, {
                "caption_only": 64, "image_only": 2, "input_delta": 64,
                "transfer_gap_change": 48, "visual_delta": 16,
            })
            # E03: model-specific neural_minus_shallow across 4 languages x 4 shallow baselines = 16.
            e03_unit = views["families"]["E03"]["units"][f"E03/{model}/comparison"]
            self.assertEqual(len(e03_unit["comparisons"]), 16)
            self.assertTrue(all(c["kind"] == "neural_minus_shallow" for c in e03_unit["comparisons"].values()))
            # E04: both poolings across 4 depths + unavailable optional arms preserved.
            self.assertEqual(views["families"]["E04"]["comparison_kinds"], {
                "caption_mean": 128, "pool_delta": 128,
            })
            unavail_arms = {a["arm"] for a in views["families"]["E04"]["unavailable_arms"]}
            self.assertEqual(unavail_arms, {"post_caption", "decoder_visual_pool"})
            # E19: cutoff lineage preserved across all 4 sources.
            for source in LANGUAGES:
                e19_unit = views["families"]["E19"]["units"][f"E19/{model}/{source}"]
                self.assertEqual(len(e19_unit["cutoffs"]), 3 * len(self.plan["categories"]) * 3)
            # E25: 4 fixed sources + diagonal, association membership hashes preserved.
            for source in (*LANGUAGES, "diagonal"):
                e25_unit = views["families"]["E25"]["units"][f"E25/{model}/{source}"]
                self.assertEqual(set(e25_unit["association_memberships"]), set(self.plan["categories"]))

            summary = result["summary"]
            self.assertEqual(summary, summarize_model_report(body))
            rendered = format_model_report_summary(summary)
            self.assertIn(f"model={model}", rendered)
            self.assertIn("post_caption", rendered)
            self.assertIn("decoder_visual_pool", rendered)
            published[model] = ref

        # Both independent model reports align cleanly in Stage 5 synthesis.
        from src.stage5.interfaces import open_release
        run = open_release(self.plan.root, self.plan.protocol, self.release)
        e03_ref = publish_owner_report(run, "e03", self.producer)
        synth_ref = publish_synthesis(run, {"qwen": published["qwen"], "gemma": published["gemma"],
                                            "e03": e03_ref}, self.producer)
        synth = self.plan.store.json(synth_ref, "body.json", SYNTHESIS_KIND)
        qwen_body = self.plan.store.json(published["qwen"], "body.json", REPORT_KIND)
        self.assertEqual(len(synth["paired_views"]),
                         sum(len(t["comparisons"]) for t in qwen_body["tables"].values()))

    def test_reuse_compatible_bundle_and_reject_mismatches(self):
        first = run_model_report(self.plan.root, self.plan.protocol, self.release, "qwen", self.producer)
        report_dir = self.plan.root / "results/stage5/reports/qwen"
        before_runs = sorted(p.name for p in report_dir.iterdir())

        reused = run_model_report(self.plan.root, self.plan.protocol, self.release, "qwen",
                                  self.producer, resume=first["reference"])
        after_runs = sorted(p.name for p in report_dir.iterdir())
        self.assertTrue(reused["resumed"])
        self.assertEqual(reused["reference"], first["reference"])
        self.assertEqual(before_runs, after_runs)

        # Reject foreign model resume.
        with self.assertRaisesRegex(ContractError, "Incompatible"):
            run_model_report(self.plan.root, self.plan.protocol, self.release, "gemma",
                             self.producer, resume=first["reference"])

        # Reject code_commit mismatch on resume.
        other_commit = dict(self.producer, code_commit="1" * 40)
        with self.assertRaisesRegex(ContractError, "code commit differs"):
            run_model_report(self.plan.root, self.plan.protocol, self.release, "qwen",
                             other_commit, resume=first["reference"])

        # Reject non-model owner selector.
        with self.assertRaisesRegex(ContractError, "MODEL must be"):
            run_model_report(self.plan.root, self.plan.protocol, self.release, "e03", self.producer)

    def test_no_numeric_prediction_reopening_for_routine_reporting(self):
        def forbid(*args, **kwargs):
            raise AssertionError("Routine Stage 5 reporting must not reopen numeric predictions")

        with patch("src.stage4.predictions.load_prediction_block", side_effect=forbid), \
             patch("src.stage4.interfaces.load_prediction_block", side_effect=forbid):
            for model in MODELS:
                res = run_model_report(self.plan.root, self.plan.protocol, self.release, model, self.producer)
                self.assertEqual(res["summary"]["unit_count"], 23)

    def test_lineage_and_matching_rule_rejections(self):
        res = run_model_report(self.plan.root, self.plan.protocol, self.release, "gemma", self.producer)
        base_body = self.plan.store.json(res["reference"], "body.json", REPORT_KIND)
        self.assertTrue(validate_model_report_body(base_body, "gemma"))

        # 1. Missing unit fails completeness.
        missing_unit = copy.deepcopy(base_body)
        missing_unit["tables"].pop("E03/gemma/comparison")
        with self.assertRaisesRegex(ContractError, "Incomplete or unexpected unit inventory"):
            validate_model_report_body(missing_unit, "gemma")

        # 2. Wrong E03 shallow route language fails matching rule.
        bad_e03 = copy.deepcopy(base_body)
        e03_table = bad_e03["tables"]["E03/gemma/comparison"]
        for comp in e03_table["comparisons"].values():
            rid = comp["inputs"][1]["route_id"]
            if bad_e03["routes"][rid]["condition"] == "word_tfidf":
                bad_e03["routes"][rid]["source_context"] = "te" if comp["view"] != "te" else "en"
                break
        with self.assertRaisesRegex(ContractError, "E03"):
            validate_model_report_body(bad_e03, "gemma")

        # 3. E19 cutoff source split violation fails matching rule.
        bad_e19 = copy.deepcopy(base_body)
        e19_table = bad_e19["tables"]["E19/gemma/en"]
        first_comp = next(iter(e19_table["comparisons"].values()))
        dev_rid = first_comp["inputs"][0]["route_id"]
        bad_e19["routes"][dev_rid]["split"] = "SCREEN"
        with self.assertRaisesRegex(ContractError, "E19 split pairing mismatch"):
            validate_model_report_body(bad_e19, "gemma")

        # 4. E25 fixed-source pairing violation fails matching rule (swap route_id to wrong source head).
        bad_e25 = copy.deepcopy(base_body)
        e25_table = bad_e25["tables"]["E25/gemma/en"]
        first_cid, first_e25 = next(iter(e25_table["comparisons"].items()))
        orig_rid = first_e25["inputs"][0]["route_id"]
        orig_route = bad_e25["routes"][orig_rid]
        wrong_rid = next(
            rid for rid, r in bad_e25["routes"].items()
            if r["block"] == orig_route["block"] and r["source_context"] == "hi"
        )
        e25_table["comparisons"][first_cid]["inputs"][0]["route_id"] = wrong_rid
        with self.assertRaisesRegex(ContractError, "E25 fixed-source pair mismatch"):
            validate_model_report_body(bad_e25, "gemma")

        # 5. Missing fit lineage on a route fails lineage check.
        bad_lineage = copy.deepcopy(base_body)
        any_rid = next(iter(bad_lineage["routes"]))
        bad_lineage["routes"][any_rid]["fit_lineage"] = {}
        with self.assertRaisesRegex(ContractError, "Missing fit lineage"):
            validate_model_report_body(bad_lineage, "gemma")

        # 6. Undeclared row comparison_id fails row integrity check.
        bad_row = copy.deepcopy(base_body)
        bad_row["tables"][f"E01/gemma/{DEPTHS['gemma'][0]}"]["rows"][0]["comparison_id"] = "undeclared"
        with self.assertRaisesRegex(ContractError, "not declared"):
            validate_model_report_body(bad_row, "gemma")

        # 7. Unaccepted Stage 4 release (pending readback marker) is rejected.
        with self.assertRaises(ContractError):
            run_model_report(self.plan.root, self.plan.protocol, self.pending_release, "gemma", self.producer)

    def test_cli_json_emission_and_notebook_contract(self):
        stdout = io.StringIO()
        with patch("src.stage5.run_model_report.current_producer", return_value=self.producer), \
             patch("src.stage5.run_model_report.read_text", return_value=json.dumps(self.plan.protocol)), \
             patch("src.stage5.run_model_report.assemble_family_views",
                   side_effect=AssertionError("CLI must summarize directly without building display views")), \
             patch("sys.stdout", stdout):
            cli_main([
                "--phase1-root", str(self.plan.root),
                "--release", json.dumps(self.release),
                "--model", "qwen",
                "--storage-scope", "local-development",
            ])
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0]["event"], "stage5_model_report_start")
        self.assertEqual(lines[0]["model"], "qwen")
        self.assertEqual(lines[-1]["status"], "passed")
        self.assertFalse(lines[-1]["stage5_complete"])
        self.assertEqual(lines[-1]["summary"]["unit_count"], 23)
        self.assertIn("path", lines[-1]["reference"])
        self.assertIn("sha256", lines[-1]["reference"])

        # Verify notebook structure and contract.
        nb_path = REPO / "scripts/stage5/run_model_report.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 4)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
        sources = ["".join(cell["source"]) for cell in nb["cells"]]
        self.assertIn("requirements-stage2-cpu.txt", sources[1])
        self.assertIn("OPENBLAS_NUM_THREADS", sources[1])
        self.assertIn("MODEL = 'qwen'", sources[2])
        self.assertIn("STAGE4_RELEASE", sources[2])
        self.assertIn("RESUME_REF", sources[2])
        self.assertIn("src.stage5.run_model_report", sources[3])
        self.assertIn("--model", sources[3])
        self.assertIn("--resume", sources[3])


if __name__ == "__main__":
    unittest.main()
