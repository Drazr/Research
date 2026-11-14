# S1-C model adapters and validation

Base: coordinator freeze `ea06d93` on `phase1`. Task branch/worktree:
`phase1-stage1-models`. No integration into phase1 is performed by this task.

`src.models.registry.create_adapter(key, config, root)` implements the frozen API.
Both selectors use the same backend and validator. The explicit model class must
be Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration (qwen), or
Gemma3ForConditionalGeneration (gemma). Other family versions are unsupported.
This is implementation support; none of these checkpoint classes has been
executed here. No checkpoint or revision has been selected or invented.

## Required configuration

Fill the shared registry's checkpoint ID, immutable model/processor/tokenizer
commit IDs, template SHA256, precision, image policy, max_input_tokens and exact
software_versions. The backend additionally requires model_class, decoder_path,
image_token_ids, processor_kwargs, device, attention_backend, repeat_atol and
repeat_rtol. Paths are explicit module paths from the selected architecture and
pinned Transformers implementation; they are never guessed from family aliases.
Changing any resolved field changes the shared configuration fingerprint.

Supported precision values are float32, bfloat16 and float16. Quantization null
means no quantization, per the frozen contract; quantized execution is unsupported.
Use image_policy `pinned_processor_with_explicit_kwargs`: the revision-pinned
processor and explicit processor_kwargs define resizing/cropping. Pixels are
decoded as RGB. There is no automatic precision, backend or template fallback.
Software versions must match installed torch, transformers, numpy and Pillow.
The validation backend requires explicitly configured eager attention. Device
choice and repeat tolerances belong in the configuration. Tolerances require
justification for the chosen precision/backend before a real run.

## Measurement behavior and unsupported sites

Only FIT ModelInput records are accepted, before image reads or processing.
Only image pixels and `Caption:\n` plus the original caption enter the native
user template; filenames, IDs, splits and targets do not. Image checksums are
verified. No generation prompt, target answer, generation or teacher forcing is
used. Inputs exceeding the token limit are rejected without truncation.

Caption alignment uses the final image-expanded IDs. A lossless decode/re-encode
with fast-tokenizer offsets must reproduce those IDs exactly; the original
caption must occur once, with complete non-whitespace coverage and no boundary
crossing into special/image/prefix tokens. Otherwise the input is unsupported,
rather than assigned a guessed span. Some tokenizer/template combinations may
fail this deliberately conservative check; their actual support is untested.
Inspection includes image-plus-caption and caption-only inputs, caption positions,
image positions, other positions, last content token and per-block measured
attention mass in both modality directions. Zero attention mass is not proof
of causal non-use. Sliding-window visibility is measured per block, not assumed
from causal token order. Large inputs may require substantial eager-attention
memory even for this smoke validation.

Decoder depths use the shared decoder_depths helper and the actual configured
decoder module count. Every selected state comes from that decoder block's
forward output, including the final block before the decoder's final norm.
`last_caption` and caption-content-only `caption_mean` are supported.
Hook handles are removed on success and failure. Evaluation mode and frozen
parameters are required for each forward. Feature batches preserve row keys,
contain finite CPU vectors, and use the shared FeatureBatch validator.

`post_caption` is unsupported pending explicit boundary semantics and validation.
Encoder/projector `visual_mean` requires explicit visual_paths entries mapping
each stage to an exposed module. Each configured output is pooled over its visual
positions independently; dimensions are not forced to match decoder dimensions.
No visual path is selected by default. Configuring a path requires checking its
meaning against the selected pinned implementation; it must expose the intended
visual stage, not a language-conditioned decoder. These features are not treated
as four language-specific observations. Actual visual availability remains untested.

The common extract API uses the default image-plus-caption condition. Caption-only
inspection checks its token spans/visibility; later caption-only feature caching
needs a coordinator-approved condition interface rather than hidden changes to
the current FeatureSite contract.

## Commands and actual validation — 15 November 2026

From this checkout:

```shell
python -m unittest discover -s tests -v
python -m compileall -q src/models scripts/stage1/validate_model.py
python scripts/stage1/validate_model.py --model qwen --phase1-root /explicit/phase1 --storage-scope local-development
python scripts/stage1/validate_model.py --model gemma --phase1-root /explicit/phase1 --storage-scope local-development
```

For real validation, add `--dataset-manifest` pointing to the passed dataset
artifact's `_COMPLETE.json`, populate the registry, install the exact chosen
versions, and use the actual persistent Drive root with
`--storage-scope persistent-drive`. The command checks all declared supported
feature sites twice and checks common decoder dimensions. It does not fit a
scientific readout. The coordinator owns end-to-end optimizer/G0 smoke checks.

Actual checks on this branch:

- **18 CPU tests passed** (11 foundation + 7 model tests), using bundled Python
  3.12 with NumPy available. Model tests cover depths, four-script span fixtures,
  invalid/ambiguous/truncated spans, non-FIT rejection before processing,
  unresolved registry rejection, measured-attention summaries and fake decoder
  block-output pooling/hook cleanup. Fake arrays/modules are software fixtures,
  not executed Qwen/Gemma checkpoints or released FIT data.
- Compilation and `git diff --check` passed.
- Both actual CLI invocations returned **blocked** (exit 2):
  `Unresolved model configuration: checkpoint_id`. They ran using system Python
  3.9 and an explicit local development root outside the worktree. No data was
  processed and no weights were loaded. Reports have separate configuration/run
  namespaces and completion checksums. These local checks do not verify Drive.
- Qwen unresolved-config namespace:
  `04a3d0288c4b3432198a6eb2fa3ae3f2de550a2dc09617f9f7dd28261a1d60fe`.
  Gemma unresolved-config namespace:
  `8246878aa3c5f73871484a90a1bbd5e747801d105e95da199107876f19273653`.
  These are raw skeleton fingerprints, not real model identities. Blocked run
  manifests keep model_fingerprint null. Checks ran with uncommitted task code;
  script hashes and code_dirty are recorded in the diagnostics.

Unresolved prerequisites: selected immutable checkpoints and processor/tokenizer
pins; template hashes; precision/image/token policies; decoder/image-token and
visual module configuration; runtime versions/device/backend/tolerances;
validated released FIT inputs and image files; installed torch/transformers;
gated-model access where applicable; actual hardware and persistent Drive checks.
Actual token alignment, attention, decoder hooks, repeatability and visual
features remain unvalidated for both checkpoints. No GPU check passed. Stage 1
and full G0 are not declared complete.

Implementation references: [Qwen2.5-VL Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)
and [Gemma 3 Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/gemma3).
These references informed the configurable loader; they do not pin a runtime.


Integration note: the coordinator selected/pinned the compact L4 starting pair and direct runtime dependencies in configs/models.json and requirements-l4.txt; see stage1_l4.md. This supersedes the earlier unselected-registry state in the historical handoff. Gemma processor metadata still requires authorized gated access. Source-configured visual paths remain untested until real validation. Model loading explicitly requires safetensors; actual model validation rejects dirty/unknown Git provenance. No checkpoint inference/GPU check has been performed by these updates.
