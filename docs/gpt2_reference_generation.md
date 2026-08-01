# GPT-2 tied-weight logits reference generation

This document records the intended semantics for `GPT2TrainingTest.LogitsConsistencyTiedWeights`.

## Training semantics

The tied-weight reference uses the corrected framework behavior where `transformer.wte.weight` and `lm_head.weight` remain the same Tensor after `model->To(CUDA)`, and `model->Parameters()` returns the shared Tensor only once.

The reference flow is:

1. Load `Data/gpt2_124M.bin`.
2. Move the model to CUDA.
3. Build an SGD optimizer from de-duplicated model parameters.
4. Run exactly 10 optimizer updates.
5. Each update starts with `ZeroGrad()`.
6. Each update uses 2 microbatches from the beginning of `tiny_shakespeare_train.bin`.
7. Each microbatch runs forward, cross entropy, and backward.
8. After 2 backward calls, run one `optimizer->Step()`.
9. After the 10th optimizer update, run one separate forward-only pass using batch offset 0.
10. Compare that final forward logits against `Data/gpt2_logits_reference_tied_10_updates.bin`.

Current gradient accumulation behavior intentionally matches the legacy test: logged loss is divided by `grad_accum_steps`, but `Backward()` uses the unscaled microbatch loss. Whether to scale gradient accumulation loss is a separate follow-up decision.

## Reference format

New tied-weight references use a fixed-width header:

```cpp
struct ReferenceHeader {
    uint32_t magic;        // 0x46523247, bytes "G2RF"
    uint32_t version;      // 1
    uint32_t dtype;        // 1 = float32
    uint32_t ndim;
    uint64_t num_elements;
};
```

The header is followed by `int64_t dims[ndim]` and then `float32 payload[num_elements]`. The reader verifies magic, version, dtype, shape product, exact file size, absence of trailing bytes, and finite payload values.

The historical `Data/gpt2_logits_reference.bin` keeps its legacy format and is only used by the disabled legacy split-weight test.

## Generation

The generator refuses to write unless explicitly enabled:

```bash
cd build/Release
TINY_GENERATE_GPT2_REFERENCE=1 \
./test_gpt2 --gtest_filter=GPT2TrainingTest.GenerateTiedWeightsReference
```

Default outputs are candidates only:

- `Data/gpt2_logits_reference_tied_10_updates.bin.generated`
- `Data/gpt2_logits_reference_tied_10_updates.meta.txt.generated`

The generator never writes the formal reference path. Candidate generation allows a dirty working tree, but metadata records `working_tree_dirty`, `unstaged_diff_sha256`, `staged_diff_sha256`, and `git_status_porcelain_sha256` so the output is not misrepresented as a clean commit artifact.

Formal promotion is intentionally not implemented in the test. Before adding it, require a clean working tree and committed generator code.

## Metadata traceability

Metadata records hashes for:

- `model_checkpoint_path` and `model_checkpoint_sha256`
- `training_dataset_path` and `training_dataset_sha256`
- `tokenizer_path` and `tokenizer_sha256`
- `training_microbatches_sha256`
- `final_forward_input_sha256`
- `final_forward_target_sha256`

`training_microbatches_sha256` covers the two repeated microbatches plus the optimizer update count, because the current loop reuses the same two microbatches for each of the 10 updates.

## Stability checks

Before promoting a generated file to the formal reference path, compare at least three generated candidates. Do not promote a reference if any pair has max absolute error greater than `1e-3`.

Use the row-level diagnostics to inspect whether mismatches are concentrated in complete `[batch, sequence]` rows:

```bash
./test_gpt2 --gtest_filter=GPT2TrainingTest.TiedReferenceCandidatePairwiseDiagnostics
```
