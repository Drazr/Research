# Selected L4 starting profile and remaining execution

Selected 15 November 2026, after the user specified L4 and no checkpoint preference. Both families remain required; run them **sequentially on a single L4** in separate Python processes. CPU tests can overlap model work. No actual L4 inference or memory measurement has been performed by this selection.

| Selector | Checkpoint | Immutable model/processor/tokenizer revision |
|---|---|---|
| qwen | Qwen/Qwen2.5-VL-3B-Instruct | 66285546d2b821cf421d4f5eb2576359d3770cd3 |
| gemma | google/gemma-3-4b-it | 093f9f388b31de276ce2de164bdc2081324b9767 |

Sources: [official Qwen model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [official Gemma model](https://huggingface.co/google/gemma-3-4b-it), [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/). Immutable revisions were retrieved from the model provider API. L4 has 24 GB nominal memory. The compact pair is a conservative resource choice, not an empirically established fit or a parameter-matched architecture comparison. Analyze each model separately before comparing patterns.

## Explicit choices

- Dense bfloat16, no quantization, eager attention for the Stage 1 visibility checks, batch size one, no generation. Eager attention can consume substantial memory; actual peak memory and failures must be recorded on the real L4.
- Maximum processed input length 1,024 tokens; inputs over the limit fail rather than truncate. Qwen image processing uses min_pixels 3,136 and max_pixels 200,704 (256 visual tokens at the configured merge factor); Gemma uses its pinned native processor with pan-and-scan disabled. Policies differ by architecture and must not be described as identical pixel preprocessing.
- Same-device, same-shape repeat checks initially require exact equality (atol=rtol=0). This is a strict diagnostic criterion, not a promise across devices or batches. Investigate a failure before proposing and documenting a tolerance change; no automatic relaxation.
- Decoder path `model.language_model.layers` and visual module paths are explicit for Transformers 4.57.6. Qwen encoder path is its final vision block (index 31), before the merger; projector path is the merger. Gemma uses the vision tower and multimodal projector. Hook availability/shapes still require real validation. The encoder reference is not claimed to be identical across architectures. `post_caption` remains unsupported pending validated boundary semantics.
- The image processor is explicitly the slow/PIL implementation (image_processor_use_fast=false); the text tokenizer remains fast for offsets. This prevents installed torchvision availability from silently changing preprocessing. torchvision 0.24.1 is pinned to match torch 2.9.1, as paired in the [PyTorch release instructions](https://pytorch.org/get-started/previous-versions/). Actual processor class and every selected package version are checked against the resolved configuration.
- Direct runtime pins are in requirements-l4.txt. Prefer a fresh Python 3.11/3.12 runtime. Transitive packages are not fully locked; record the complete installed environment for each run. No runtime versions are silently inferred during inference.

The configured module paths were checked against the publisher's tagged [Qwen implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py) and [Gemma implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/gemma3/modeling_gemma3.py). This is source inspection, not executed checkpoint validation.

## Access and runtime configuration

Gemma is gated. The current unauthenticated metadata request returned HTTP 401 for its config, processor and tokenizer files. The user must have appropriate Hugging Face access and complete any required license/access steps themselves. Use a normal Hugging Face login or a runtime secret; never put a token in Git, configuration files, reports or chat. No gated metadata was guessed or bypassed. Accordingly the repository's Gemma template hash/image-token fields remain null until authorized resolution.

`src/stage1/configure_l4.py` checks the actual GPU and package versions, fetches pinned processor/tokenizer/config metadata, resolves template hash and image tokens, and publishes a per-model configuration artifact. It does **not** download model weights or run inference. A passed configuration report establishes configuration readiness only. Both validators must still pass afterward. Keep generated runtime configuration outside Git and pass its printed `models.json` path to subsequent commands.

## Persistent storage check

The current Windows machine has no PHASE1_ROOT or verified synced Drive folder. The previously connected Drive account had no phase1 folder, and the user chose to connect a different account. No folder/artifact was created in the old account. Connection to the intended Drive and the mounted filesystem remain prerequisites.

No hard links, atomic renames or shared canonical destination writes are required by the updated source downloader. Each source cache is a unique, completion-verified artifact under `cache/shared/source-downloads/<source_sha256>/<run_id>/`. Reuse verifies payload hashes. Legacy cache files may be read only after their pinned hash matches and are copied without overwriting them. Partial artifacts are not consumed; corrupted completed artifacts fail. Filesystem tests with hard links disabled passed locally. This does **not** verify a real Drive mount or cross-client synchronization.

In the actual runtime, use dataset preparation/validation to exercise the real source cache and artifact protocol. For a small first filesystem test, create a temporary-directory base under PHASE1_ROOT and run the cache/foundation tests with TMPDIR pointing there. Inspect the test result; a successful mount or folder listing alone is insufficient evidence of durable artifact behavior. No dataset/model download is implied by the following software tests.

## Colab example (only if Colab is the chosen L4 host)

1. Select an L4 GPU runtime. Mount the intended Google Drive using Colab's normal user sign-in flow:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Clone the repository's phase1 branch to local runtime storage and install the selected environment before importing model packages. Restart if an incompatible already-imported package requires it. Do not place the live Git checkout in Drive.

```bash
git clone --branch phase1 https://github.com/Drazr/Research.git /content/Research
cd /content/Research
python -m pip install -r requirements-l4.txt
export PHASE1_ROOT=/content/drive/MyDrive/phase1
mkdir -p "$PHASE1_ROOT/cache/storage-tests"
TMPDIR="$PHASE1_ROOT/cache/storage-tests" python -m unittest discover -s tests -p 'test_stage1_cache.py' -v
TMPDIR="$PHASE1_ROOT/cache/storage-tests" python -m unittest discover -s tests -p 'test_stage1_foundation.py' -v
```

3. Prepare/validate the data on that actual storage using the actual checkout commit. Preserve the printed dataset-manifest path for the next commands.

```bash
cd /content/Research
python -m src.stage1.prepare_dataset --phase1-root "$PHASE1_ROOT" --code-commit "$(git rev-parse HEAD)" --storage-scope persistent-drive
# Set DATASET_MANIFEST to the printed absolute _COMPLETE.json path.
python -m src.stage1.validate_data --phase1-root "$PHASE1_ROOT" --dataset-manifest "$DATASET_MANIFEST" --code-commit "$(git rev-parse HEAD)" --storage-scope persistent-drive
```

4. For **one model at a time**, set MODEL=qwen, then repeat with MODEL=gemma. Configuration resolution needs authorized model access, the passed dataset artifact and the actual L4. Set MODEL_CONFIG to that command's printed models_config path; do not accidentally use the other model's unconfigured registry.

```bash
python -m src.stage1.configure_l4 --model "$MODEL" --phase1-root "$PHASE1_ROOT" --dataset-manifest "$DATASET_MANIFEST" --code-commit "$(git rev-parse HEAD)" --storage-scope persistent-drive
# Set MODEL_CONFIG to the printed absolute models.json path.
python -m src.stage1.validate_model --model "$MODEL" --phase1-root "$PHASE1_ROOT" --dataset-manifest "$DATASET_MANIFEST" --models-config "$MODEL_CONFIG" --storage-scope persistent-drive
python -m src.stage1.run_validity_smoke --model "$MODEL" --phase1-root "$PHASE1_ROOT" --dataset-manifest "$DATASET_MANIFEST" --models-config "$MODEL_CONFIG" --code-commit "$(git rev-parse HEAD)" --storage-scope persistent-drive --batch-size 1
```

Run from a clean checkout. Model validation now rejects dirty/unknown Git provenance before accepting an actual pass and records hashes of adapter/shared source files. Do not run the later commands after a blocked/failed prior command. Report token alignment, visibility, site availability, actual device/precision, memory failures and repeated-feature checks honestly. Stage 1 only completes after real data/Drive/model checks for both models pass and the coordinator reviews the evidence. Full G0 controls and scientific experiments remain later stages.


## Post-integration local checks

At merge commit 4030434c90183abd3c2b8416bbd0dec22b632335, the real-data prepare/resume command passed using the available local CPU environment, migrating the legacy pinned sources to the new completion-verified cache and returning the same dataset identity. Model/runtime requirements were not installed in this CPU environment. A runtime-configuration preflight exposed a Windows MAX_PATH failure when writing nested hash directories; configuration artifacts now use the shorter `cache/runtime-configurations/<model>/<run_id>` namespace, retaining full dataset/protocol/model fingerprints in their completion manifest. This is a configuration cache, not a fitted-model checkpoint. A regression test checks persistence of a blocked report under a long development root.

The combined software suite now has 29 passing tests. CUDA availability and bfloat16 support fail before processor/weight downloads when the selected device is absent. Model validators/smoke reports record observed GPU peak allocation/reservation only when actual CUDA information is available; these metrics are not invented from parameter counts.


Notebook entry points (user-requested conversion): `scripts/stage1/stage1_cpu.ipynb` runs preparation then validation; `configure_l4.ipynb`, `validate_model.ipynb` and `run_validity_smoke.ipynb` run in that order per MODEL. The five former scripts/stage1 Python entry files are removed. Their shared implementations now live in src/stage1; scientific rules, artifacts and completion gates are unchanged. Notebook syntax and CPU fixture checks do not establish actual Colab/L4/Drive completion.
