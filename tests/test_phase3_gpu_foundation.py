"""Unit and contract tests for Phase 3 GPU Track Stage B1 foundation."""

import json
import sys
import unittest

from src.common.contracts import ContractError, LANGUAGES, ModelInput, TargetRecord
from src.phase3.interfaces import (
    E23_CHUNK_SIZE_PAIRS,
    E31_SITES,
    E39_E31_CHUNK_SIZE_IMAGES,
    E39_SITES,
    KIND_GPU_PLAN,
    PHASE3_GPU_DECISION_RANGES,
    PHASE3_GPU_VERSION,
    REPO,
    SCHEMA_GPU_PLAN,
    build_e23_pair_plan,
    select_e39_token_indices,
)
import src.phase3.gpu_plan as gpu_plan


class Phase3GpuFoundationTests(unittest.TestCase):
    def test_lazy_imports_and_constants(self):
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("scipy", sys.modules)
        self.assertEqual(PHASE3_GPU_VERSION, "phase3-gpu-stageB1.v1")
        self.assertEqual(KIND_GPU_PLAN, "phase3_gpu_plan")
        self.assertEqual(SCHEMA_GPU_PLAN, "phase3.gpu_plan.v1")
        self.assertEqual(len(E39_SITES), 8)
        self.assertEqual(len(E31_SITES), 3)
        self.assertEqual(set(PHASE3_GPU_DECISION_RANGES.keys()), {"E39", "E31", "E23"})

    def test_select_e39_token_indices_ascii_indic_and_fallback(self):
        res_en = select_e39_token_indices(["A", "brown", "dog", "runs", "."])
        self.assertEqual(res_en["first_caption"], 0)
        self.assertEqual(res_en["middle_caption"], 2)
        self.assertEqual(res_en["penultimate_caption"], 3)
        self.assertEqual(res_en["last_non_punct_caption"], 3)
        self.assertFalse(res_en["last_non_punct_fallback_used"])
        self.assertEqual(res_en["last_caption"], 4)
        self.assertEqual(res_en["first_half_span"], [0, 2])
        self.assertEqual(res_en["second_half_span"], [2, 5])

        # Hindi danda + whitespace trailing tokens must be skipped by last_non_punct_caption
        res_hi = select_e39_token_indices(["एक", "कुत्ता", "।", " "])
        self.assertEqual(res_hi["last_non_punct_caption"], 1)
        self.assertFalse(res_hi["last_non_punct_fallback_used"])

        # All-punctuation fallback
        res_punct = select_e39_token_indices([".", ",", "!"])
        self.assertEqual(res_punct["last_non_punct_caption"], 2)
        self.assertTrue(res_punct["last_non_punct_fallback_used"])

        with self.assertRaises(ContractError):
            select_e39_token_indices([])

    def test_e23_pair_plan_and_e39_e31_plan(self):
        protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        cats = list(protocol["category_order"])
        prim = list(protocol["primary_categories"])
        plant_idx = cats.index("plant.n.02")

        screen_ids = [f"{i:016x}" for i in range(60)]
        screen_ids[0] = "6a20677eee04fd08"

        screen_targets = {}
        draw0_records = {}
        for i, iid in enumerate(screen_ids):
            vals = [1 if (i % 2 == 0) else 0 for _ in cats]
            if iid == "6a20677eee04fd08":
                vals[plant_idx] = None
            screen_targets[iid] = TargetRecord(iid, "SCREEN", tuple(vals))
            for lang in LANGUAGES:
                draw0_records[(iid, lang)] = ModelInput(
                    image_id=iid,
                    image_path=f"images/{iid}.jpg",
                    image_sha256="a" * 64,
                    language=lang,
                    caption_id=f"{iid}_{lang}_0",
                    caption_text=f"caption {iid} {lang}",
                    split="SCREEN",
                )

        p1 = build_e23_pair_plan(screen_ids, screen_targets, draw0_records, cats, prim, 20)
        p2 = build_e23_pair_plan(list(reversed(screen_ids)), screen_targets, draw0_records, cats, prim, 20)
        self.assertEqual(p1, p2)
        self.assertEqual(p1["confirm_rows"], 0)
        self.assertTrue(p1["zero_model_scores_in_pairing"])
        # Unknown cell 6a20677eee04fd08 on plant.n.02 has 1 unknown, 29 pos, 30 neg -> 29 disjoint pairs
        self.assertEqual(p1["per_category"]["plant.n.02"]["n_unknown"], 1)
        self.assertEqual(p1["per_category"]["plant.n.02"]["n_pairs"], 29)
        self.assertEqual(p1["per_category"]["plant.n.02"]["support_status"], "ok")
        for chunk in p1["chunks"]:
            self.assertEqual(chunk["split"], "SCREEN")
            self.assertLessEqual(chunk["n_pairs"], E23_CHUNK_SIZE_PAIRS)

        ep = gpu_plan.build_e39_e31_extraction_plan(screen_ids, draw0_records)
        self.assertEqual(ep["confirm_rows"], 0)
        self.assertEqual(ep["total_images"], 60)
        self.assertEqual(ep["views_per_condition"], 240)
        self.assertEqual(ep["total_chunks"], 2)
        self.assertEqual(ep["chunks"][0]["n_images"], E39_E31_CHUNK_SIZE_IMAGES)

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/prepare_phase3_gpu.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 4)
        full = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("CODE_REF = 'phase3'", full)
        self.assertIn("src.phase3.gpu_plan", full)
        self.assertIn("93cbf480e9d8c13401aaa98315444f3a4d2b3b57ec769b458a799dc269c40439", full)


if __name__ == "__main__":
    unittest.main()
