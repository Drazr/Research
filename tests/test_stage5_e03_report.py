"""Fixture tests for the shared Stage 5 E03 reporting worker and CPU entry point."""

import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError, LANGUAGES
from src.stage2.schema import REPO, plain
from src.stage4.interfaces import MACRO, METRICS
from src.stage4.validation import PENDING, RELEASE, _publish
from src.stage5.interfaces import REPORT_KIND, open_release, publish_owner_report
from src.stage5.run_e03_report import (
    EXPECTED_COMPARISON_COUNT,
    EXPECTED_ROUTE_COUNT,
    OWNER,
    SHARED_UNIT_ID,
    build_shallow_reference_rows,
    format_e03_summary,
    main,
    run_e03_report,
    verify_e03_report,
)

import test_stage4_foundation as stage4


def _build_full_e03_rows(plan, unit):
    """Build deterministic full-coverage rows covering ok, low-support, undefined and intervals."""
    rows = []
    categories = list(plan["categories"])
    total_draws = plan["bootstrap"]["draws"]
    for c_idx, (cid, comp) in enumerate(sorted(unit["comparisons"].items())):
        is_macro_comp = comp["kind"] == "pooled_context_mean"
        for m_idx, metric in enumerate(comp["statistics"]):
            for cat_idx, category in enumerate(categories):
                support = {"known": 10, "positive": 4, "negative": 6}
                if is_macro_comp:
                    support["views"] = len(LANGUAGES)
                if cat_idx == 0 and m_idx == 0:
                    row = {
                        "comparison_id": cid, "category": category, "statistic": metric,
                        "value": None, "status": "undefined", "reason": "single_class_in_screen",
                        "support": {"known": 10, "positive": 0, "negative": 10,
                                    **({"views": len(LANGUAGES)} if is_macro_comp else {})},
                        "interval": {"lower": None, "upper": None, "valid_draws": 0, "total_draws": total_draws},
                    }
                elif cat_idx == 1 and m_idx == 0:
                    val = round(0.45 + 0.001 * c_idx, 6)
                    row = {
                        "comparison_id": cid, "category": category, "statistic": metric,
                        "value": val, "status": "descriptive_low_support",
                        "reason": "screen_support_below_minimum_5",
                        "support": support,
                        "interval": {"lower": round(val - 0.05, 6), "upper": round(val + 0.05, 6),
                                     "valid_draws": total_draws, "total_draws": total_draws},
                    }
                else:
                    val = round(0.55 + 0.001 * ((c_idx + m_idx + cat_idx) % 40), 6)
                    row = {
                        "comparison_id": cid, "category": category, "statistic": metric,
                        "value": val, "status": "ok", "reason": None,
                        "support": support,
                        "interval": {"lower": round(val - 0.04, 6), "upper": round(val + 0.04, 6),
                                     "valid_draws": total_draws, "total_draws": total_draws},
                    }
                rows.append(row)
            macro_val = round(0.60 + 0.001 * ((c_idx + m_idx) % 20), 6)
            rows.append({
                "comparison_id": cid, "category": MACRO, "statistic": metric,
                "value": macro_val, "status": "ok", "reason": None,
                "support": {"categories": len(plan["primary"]), "estimable": len(plan["primary"])},
                "interval": {"lower": round(macro_val - 0.03, 6), "upper": round(macro_val + 0.03, 6),
                             "valid_draws": total_draws, "total_draws": total_draws},
            })
    return rows


class Stage5E03ReportTests(unittest.TestCase):
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
            "schema": "stage4.release.v1", "plan": plain(plan.reference), "plan_identity": plan["identity"],
            "dataset_fingerprint": plan["dataset_fingerprint"], "stage3_registry": plain(plan["registry"]),
            "g0_gate": plain(plan["g0_gate"]), "stage4_complete": False, "optional_coverage": [],
            "analyses": {uid: stub("analysis", uid) for uid in units},
            "predictions": {bid: stub("prediction", bid) for bid in plan["blocks"]},
        }

        def write_release(kind, payload):
            return _publish(plan.store, plan["dataset_fingerprint"], "fixture/stage5-e03", kind,
                            payload, {}, [plan.reference], cls.producer)

        cls.pending_release = write_release(PENDING, body)
        cls.release = write_release(
            RELEASE, dict(body, stage4_complete=True, independent_readback=cls.pending_release),
        )
        cls.fixture_analysis = {}
        for uid, unit in units.items():
            if uid == SHARED_UNIT_ID:
                rows = _build_full_e03_rows(plan, unit)
            else:
                cid = next(iter(unit["comparisons"]))
                rows = [{"comparison_id": cid, "category": MACRO,
                         "statistic": unit["comparisons"][cid]["statistics"][0], "value": 0.5,
                         "status": "ok", "reason": None, "support": {"known": 4}}]
            cls.fixture_analysis[body["analyses"][uid]["path"]] = {
                "unit_id": uid,
                "predictions": {bid: body["predictions"][bid] for bid in unit["required_blocks"]},
                "rows": rows,
                "cutoffs": [],
                "association_memberships": {},
            }

        def read_artifact(plan_arg, reference, *, units):
            return cls.fixture_analysis[reference["path"]]

        patcher = patch("src.stage5.interfaces.load_analysis_block", side_effect=read_artifact)
        patcher.start()
        cls.addClassCleanup(patcher.stop)

    def test_exact_e03_coverage_and_display_projection(self):
        events = []
        with patch("src.stage5.run_e03_report.verify_e03_report", wraps=verify_e03_report) as verify_spy, \
             patch("src.stage5.run_e03_report.build_shallow_reference_rows",
                   side_effect=AssertionError("CLI summary should not materialize sorted display rows")):
            result = run_e03_report(
                self.plan.root, self.plan.protocol, self.release, self.producer, progress=events.append,
            )
        self.assertEqual(verify_spy.call_count, 1)
        ref = result["reference"]
        self.assertTrue(ref["path"].startswith("results/stage5/reports/e03/"))
        self.assertTrue((self.plan.root / ref["path"]).is_file())
        self.assertEqual([e["event"] for e in events], ["report_unit_read", "stage5_e03_report_verified"])

        body = self.plan.store.json(ref, "body.json", REPORT_KIND)
        self.assertEqual(body["owner"], OWNER)
        self.assertEqual(set(body["tables"]), {SHARED_UNIT_ID})
        self.assertEqual(body["optional_coverage"], {})

        expected_rows = EXPECTED_COMPARISON_COUNT * (len(self.plan["categories"]) + 1) * len(METRICS)
        summary = result["summary"]
        self.assertEqual(summary["coverage"]["routes"], EXPECTED_ROUTE_COUNT)
        self.assertEqual(summary["coverage"]["predictions"], EXPECTED_ROUTE_COUNT)
        self.assertEqual(summary["coverage"]["comparisons"], EXPECTED_COMPARISON_COUNT)
        self.assertEqual(summary["coverage"]["rows"], expected_rows)
        self.assertEqual(summary["coverage"]["interval_rows"], expected_rows)
        self.assertEqual(summary["coverage"]["status_counts"]["undefined"], EXPECTED_COMPARISON_COUNT)
        self.assertEqual(summary["coverage"]["status_counts"]["descriptive_low_support"], EXPECTED_COMPARISON_COUNT)

        per_comp_rows = (len(self.plan["categories"]) + 1) * len(METRICS)
        self.assertEqual(summary["rows_by_scope"], {
            "language_specific": 12 * per_comp_rows,
            "pooled_context_macro": 1 * per_comp_rows,
            "pooled_per_context": 4 * per_comp_rows,
        })
        self.assertEqual(summary["rows_by_recipe"], {
            "char_tfidf": 4 * per_comp_rows,
            "length": 4 * per_comp_rows,
            "pooled_char_tfidf": 5 * per_comp_rows,
            "word_tfidf": 4 * per_comp_rows,
        })

        display = build_shallow_reference_rows(body)
        self.assertEqual(len(display), expected_rows)
        self.assertTrue(all(row["kind"] != "neural_minus_shallow" for row in display))
        macro_view_rows = [row for row in display if row["scope"] == "pooled_context_macro"]
        self.assertEqual(len(macro_view_rows), per_comp_rows)
        self.assertTrue(all(row["views"] == list(LANGUAGES) and row["recipe"] == "pooled_char_tfidf"
                            for row in macro_view_rows))
        self.assertTrue(all(row["analysis_reference"] == body["tables"][SHARED_UNIT_ID]["reference"]
                            and row["prediction_reference"]["path"].startswith("fixture/prediction/")
                            and row["fit_lineage"] for row in display))

        # Verify no second canonical table cache was created alongside body.json.
        report_dir = (self.plan.root / ref["path"]).parent
        self.assertEqual(sorted(p.name for p in report_dir.iterdir()), ["_COMPLETE.json", "body.json"])

    def test_resume_and_incompatible_resume_rejection(self):
        first = run_e03_report(self.plan.root, self.plan.protocol, self.release, self.producer)
        resumed = run_e03_report(
            self.plan.root, self.plan.protocol, self.release, self.producer, resume=first["reference"],
        )
        self.assertEqual(resumed["reference"], first["reference"])

        other_commit = dict(self.producer, code_commit="f" * 40)
        with self.assertRaisesRegex(ContractError, "code commit differs"):
            run_e03_report(self.plan.root, self.plan.protocol, self.release, other_commit, resume=first["reference"])

        run = open_release(self.plan.root, self.plan.protocol, self.release)
        qwen_ref = publish_owner_report(run, "qwen", self.producer)
        with self.assertRaisesRegex(ContractError, "Incompatible"):
            run_e03_report(self.plan.root, self.plan.protocol, self.release, self.producer, resume=qwen_ref)

    def test_reject_incompatible_inputs_and_corrupted_reports(self):
        with self.assertRaises(ContractError):
            run_e03_report(self.plan.root, self.plan.protocol, self.pending_release, self.producer)

        bad_sha = dict(self.release, sha256="1" * 64)
        with self.assertRaises(ContractError):
            run_e03_report(self.plan.root, self.plan.protocol, bad_sha, self.producer)

        valid_ref = run_e03_report(self.plan.root, self.plan.protocol, self.release, self.producer)["reference"]
        valid_body = self.plan.store.json(valid_ref, "body.json", REPORT_KIND)

        # Wrong owner
        wrong_owner = copy.deepcopy(valid_body)
        wrong_owner["owner"] = "qwen"
        with self.assertRaisesRegex(ContractError, "not shared E03"):
            verify_e03_report(wrong_owner)

        # Extra neural-minus-shallow model unit injected
        extra_unit = copy.deepcopy(valid_body)
        extra_unit["tables"]["E03/qwen/comparison"] = {"rows": []}
        with self.assertRaisesRegex(ContractError, "only the accepted E03/e03/shared unit"):
            verify_e03_report(extra_unit)

        # neural_minus_shallow comparison kind injected into shared table
        neural_comp = copy.deepcopy(valid_body)
        first_cid = next(iter(neural_comp["tables"][SHARED_UNIT_ID]["comparisons"]))
        neural_comp["tables"][SHARED_UNIT_ID]["comparisons"][first_cid]["kind"] = "neural_minus_shallow"
        with self.assertRaisesRegex(ContractError, "model workers"):
            verify_e03_report(neural_comp)

        # Missing pooled context macro comparison
        missing_macro = copy.deepcopy(valid_body)
        macro_cid = next(
            cid for cid, comp in missing_macro["tables"][SHARED_UNIT_ID]["comparisons"].items()
            if comp["kind"] == "pooled_context_mean"
        )
        del missing_macro["tables"][SHARED_UNIT_ID]["comparisons"][macro_cid]
        missing_macro["tables"][SHARED_UNIT_ID]["rows"] = [
            row for row in missing_macro["tables"][SHARED_UNIT_ID]["rows"]
            if row["comparison_id"] != macro_cid
        ]
        with self.assertRaisesRegex(ContractError, "comparison count mismatch"):
            verify_e03_report(missing_macro)

        # Pooled context macro missing a language
        partial_macro = copy.deepcopy(valid_body)
        partial_macro["tables"][SHARED_UNIT_ID]["comparisons"][macro_cid]["views"] = list(LANGUAGES[:-1])
        with self.assertRaisesRegex(ContractError, "all four scheduled languages"):
            verify_e03_report(partial_macro)

        # Missing row
        missing_row = copy.deepcopy(valid_body)
        missing_row["tables"][SHARED_UNIT_ID]["rows"].pop()
        with self.assertRaisesRegex(ContractError, "row count mismatch"):
            verify_e03_report(missing_row)

        # Duplicate row
        dup_row = copy.deepcopy(valid_body)
        dup_row["tables"][SHARED_UNIT_ID]["rows"][-1] = copy.deepcopy(
            dup_row["tables"][SHARED_UNIT_ID]["rows"][0],
        )
        with self.assertRaisesRegex(ContractError, "Unexpected or duplicate"):
            verify_e03_report(dup_row)

        # Non-empty optional_coverage or cutoffs
        with_opt = copy.deepcopy(valid_body)
        with_opt["optional_coverage"] = {"post_caption": "unavailable"}
        with self.assertRaisesRegex(ContractError, "optional coverage"):
            verify_e03_report(with_opt)

    def test_cli_entry_point_and_notebook_contract(self):
        stdout = io.StringIO()
        protocol_text = json.dumps(self.plan.protocol)
        with patch("src.stage5.run_e03_report.current_producer", return_value=self.producer), \
             patch("src.stage5.run_e03_report._read_text", return_value=protocol_text), \
             patch("sys.stdout", stdout):
            result = main([
                "--phase1-root", str(self.plan.root),
                "--release", json.dumps(self.release),
                "--storage-scope", "local-development",
            ])
        lines = [json.loads(line) for line in stdout.getvalue().strip().splitlines()]
        self.assertEqual(lines[0]["event"], "stage5_e03_report_start")
        self.assertEqual(lines[-1]["status"], "passed")
        self.assertEqual(lines[-1]["owner"], OWNER)
        self.assertFalse(lines[-1]["stage5_complete"])
        self.assertEqual(lines[-1]["reference"], result["reference"])

        # CLI resume returns the exact same reference
        resume_stdout = io.StringIO()
        with patch("src.stage5.run_e03_report.current_producer", return_value=self.producer), \
             patch("src.stage5.run_e03_report._read_text", return_value=protocol_text), \
             patch("sys.stdout", resume_stdout):
            resumed = main([
                "--phase1-root", str(self.plan.root),
                "--release", json.dumps(self.release),
                "--resume", json.dumps(result["reference"]),
                "--storage-scope", "local-development",
            ])
        self.assertEqual(resumed["reference"], result["reference"])
        resume_lines = [json.loads(line) for line in resume_stdout.getvalue().strip().splitlines()]
        self.assertTrue(resume_lines[0]["resume"])
        self.assertEqual(resume_lines[-1]["status"], "passed")

        # Reject model selector and malformed release JSON
        with self.assertRaises(SystemExit):
            main(["--phase1-root", str(self.plan.root), "--release", json.dumps(self.release), "--model", "qwen"])
        with self.assertRaisesRegex(ContractError, "Invalid JSON"):
            main(["--phase1-root", str(self.plan.root), "--release", "not-json"])

        # Notebook structure contract
        notebook_path = REPO / "scripts/stage5/run_e03_report.ipynb"
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 4)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
        setup_src = "".join(nb["cells"][1]["source"])
        settings_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("checkout','--detach'", setup_src)
        self.assertIn("requirements-stage2-cpu.txt", setup_src)
        self.assertIn("STAGE4_RELEASE", settings_src)
        self.assertNotIn("MODEL", settings_src)
        self.assertIn("src.stage5.run_e03_report", exec_src)
        self.assertIn("persistent-drive", exec_src)
        self.assertNotIn("--model", exec_src)


if __name__ == "__main__":
    unittest.main()
