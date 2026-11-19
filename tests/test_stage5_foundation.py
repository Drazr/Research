"""Stage 5 coordinator fixtures on the real Stage 4 workload inventory."""

import unittest
from unittest.mock import patch

from src.common.contracts import ContractError
from src.stage2.schema import plain
from src.stage4.validation import PENDING, RELEASE, _publish
from src.stage5.coordinator import FINAL_KIND, SYNTHESIS_KIND, publish_final_report, publish_synthesis
from src.stage5.interfaces import OWNERS, REPORT_KIND, open_release, publish_owner_report

import test_stage4_foundation as stage4


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the synthetic accepted Stage 3 registry and genuine 47-unit inventory.
        fixture = stage4.AcceptedInputTests
        fixture.setUpClass()
        cls.addClassCleanup(fixture.doClassCleanups)
        cls.plan = plan = fixture.plan
        cls.producer = dict(fixture.fixture.producer, storage_scope="local-development")
        units = stage4.analysis_units(plan)

        def stub(prefix, key):
            return {"path": f"fixture/{prefix}/{key}/_COMPLETE.json", "sha256": "0" * 64}

        body = {
            "schema": "stage4.release.v1", "plan": plain(plan.reference), "plan_identity": plan["identity"],
            "dataset_fingerprint": plan["dataset_fingerprint"], "stage3_registry": plain(plan["registry"]),
            "g0_gate": plain(plan["g0_gate"]), "stage4_complete": False, "optional_coverage": [],
            "analyses": {uid: stub("analysis", uid) for uid in units},
            "predictions": {bid: stub("prediction", bid) for bid in plan["blocks"]},
        }

        def write_release(kind, body):
            return _publish(plan.store, plan["dataset_fingerprint"], "fixture/stage5", kind,
                            body, {}, [plan.reference], cls.producer)

        pending = write_release(PENDING, body)
        cls.release = write_release(RELEASE, dict(body, stage4_complete=True, independent_readback=pending))
        cls.report_run = open_release(plan.root, plan.protocol, cls.release)
        cls.fixture_analysis = {}
        # Stub the numerical artifacts. Stage 4 tests own their exhaustive row validation.
        for uid, unit in units.items():
            cid = next(iter(unit["comparisons"]))
            cls.fixture_analysis[body["analyses"][uid]["path"]] = {
                "unit_id": uid, "predictions": {bid: body["predictions"][bid] for bid in unit["required_blocks"]},
                "rows": [{"comparison_id": cid, "category": "primary_macro",
                          "statistic": unit["comparisons"][cid]["statistics"][0], "value": .5,
                          "status": "ok", "support": {"known": 4}}],
                "cutoffs": [], "association_memberships": {},
            }

        def read_artifact(plan, reference, *, units):
            return cls.fixture_analysis[reference["path"]]

        patcher = patch("src.stage5.interfaces.load_analysis_block", side_effect=read_artifact)
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        cls.reports = {owner: publish_owner_report(cls.report_run, owner, cls.producer) for owner in OWNERS}

    def test_full_inventory_resume_and_foreign_owner(self):
        observed = {owner: self.plan.store.json(ref, "body.json", REPORT_KIND)
                    for owner, ref in self.reports.items()}
        self.assertEqual([len(observed[o]["tables"]) for o in OWNERS], [23, 23, 1])
        self.assertEqual(publish_owner_report(self.report_run, "qwen", self.producer, resume=self.reports["qwen"]),
                         self.reports["qwen"])
        with self.assertRaisesRegex(ContractError, "Incompatible"):
            publish_owner_report(self.report_run, "gemma", self.producer, resume=self.reports["qwen"])
        with self.assertRaisesRegex(ContractError, "Incomplete|Require both"):
            publish_synthesis(self.report_run, {"qwen": self.reports["qwen"]}, self.producer)

    def test_synthesis_and_evidence_readback(self):
        ref = publish_synthesis(self.report_run, self.reports, self.producer)
        body = self.plan.store.json(ref, "body.json", SYNTHESIS_KIND)
        self.assertFalse(body["stage5_complete"])
        qwen = self.plan.store.json(self.reports["qwen"], "body.json", REPORT_KIND)
        self.assertEqual(len(body["paired_views"]), sum(len(t["comparisons"]) for t in qwen["tables"].values()))
        uid, table = next(iter(qwen["tables"].items()))
        row = table["rows"][0]
        evidence = {k: row[k] for k in ("comparison_id", "category", "statistic")}
        evidence["unit"] = uid
        record = {
            "id": "fixture", "question": "Q6", "scope": "Fixture model and cohort", "finding": "Fixture finding",
            "disposition": "revise", "alternatives": "Fixture alternative", "next_uncertainty": "What varies",
            "decision_change": "Inspect matching cohorts", "next_test": {"test": "E11", "dependencies": "Fixture"},
            "evidence": [evidence],
        }
        review = {"summary": "Fixture review", "decisions": [record]}
        final = publish_final_report(self.report_run, ref, review, self.producer)
        body = self.plan.store.json(final, "body.json", FINAL_KIND)
        self.assertFalse(body["stage5_complete"])
        self.assertEqual(body["review"]["decisions"][0]["evidence_rows"][0]["analysis"], table["reference"])
        with self.assertRaisesRegex(ContractError, "persistent Drive"):
            publish_final_report(self.report_run, ref, review, dict(self.producer, storage_scope="persistent-drive"))
        evidence["statistic"] = "fabricated"
        with self.assertRaisesRegex(ContractError, "does not resolve"):
            publish_final_report(self.report_run, ref, review, self.producer)
