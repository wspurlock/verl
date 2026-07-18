# Phase 3: Positive-Example Language-Model Loss

**Prerequisite:** Phases 1 and 2 landed and passed acceptance tests.  
**Scope:** correctness provenance, transfer-queue plumbing, auxiliary actor loss, global normalization, metrics, and tests.  
**Out of scope:** changing reward functions or adding a second model forward.

## 1. Objective

For verifier-confirmed correct sampled responses, add a token-level negative log-likelihood term to the existing PPO actor loss:

```text
L_actor = L_PPO + positive_lm_coef * L_positive_lm
```

Paper default:

```text
positive_lm_coef = 0.1
```

Reuse current-policy response-token log probabilities already produced during the actor training forward. Do not run an additional actor forward solely for this loss.

## 2. Correctness contract

Configuration:

```yaml
positive_lm:
  enabled: false
  coef: 0.1
  correctness_key: acc
  correctness_threshold: 1.0
```

A sequence is positive when:

```text
float(verifier_output[correctness_key]) >= correctness_threshold
```

Initial implementation supports numeric scalar correctness fields only. Boolean values may be accepted as 0/1. Reject strings, arrays, missing values, NaN, and infinity.

The correctness field must originate from reward/verifier extra information and be carried alongside the trajectory through the v1 transfer queue. It must not be inferred from:

- KL-shaped `token_level_rewards`;
- response length penalties;
- the actor advantage;
- total reward when that total mixes correctness with other terms;
- whether the PPO advantage is positive.

If positive LM loss is enabled and any non-padding trajectory lacks a valid configured correctness value, fail the batch with an error identifying the key and reward source. Do not silently treat missing correctness as negative.

Balancing-only padding samples are excluded and need no correctness value.

## 3. Loss

For current actor log probabilities `log_pi[i,t]`:

```text
positive_mask[i,t] =
    is_positive[i] * response_mask[i,t] * real_sample_mask[i]

local_numerator = -sum(positive_mask * log_pi)
local_denominator = sum(positive_mask)
```

The effective numerator and denominator must be global across the actor data-parallel group and the configured global PPO minibatch.

Requirements:

- identical scale regardless of DP world size;
- invariant to microbatch splitting and gradient accumulation;
- no prompt or padding tokens;
- balancing padding excluded;
- all valid response tokens of a positive trajectory included;
- zero global positives yields an exactly zero differentiable loss;
- the denominator is detached;
- no division by an epsilon that creates an arbitrary nonzero scale for empty batches;
- compatible with mixed precision without accumulating counts in low precision.

Do not clip the positive LM loss through the PPO ratio. Its coefficient is the only direct scale parameter.

## 4. PPO-loss integration

Add the auxiliary loss in the shared actor loss layer used by v1 FSDP2. Keep the registered vanilla PPO loss responsible for the PPO surrogate; compose the auxiliary term around it rather than duplicating the surrogate implementation.

Preserve:

- asymmetric clip configuration;
- rollout correction weights on the PPO term only;
- entropy regularization semantics;
- existing KL-loss semantics;
- existing actor metrics.

The positive LM term is based on the current actor's response-token log probabilities, not old or rollout log probabilities.

When disabled or when `coef=0`, avoid correctness-field requirements and preserve the original actor loss path.

## 5. Transfer-queue plumbing

Carry a compact per-sequence correctness value or boolean positive flag from reward computation through actor update. Prefer storing the raw configured correctness scalar until thresholding at a well-defined preparation boundary, so diagnostics can audit the threshold.

Requirements:

- repeated trajectories retain their individual correctness values;
- prompt UID grouping does not collapse correctness values;
- batch reorder/balance preserves alignment;
- padding introduced for divisibility is distinguishable;
- checkpoint/resume does not require serializing transient batch correctness fields;
- the actor worker receives the field in every minibatch when enabled.

## 6. Metrics

Log:

```text
actor/positive_lm_loss
actor/positive_lm_weighted_loss
actor/positive_lm_coef
actor/positive_sequence_count
actor/positive_sequence_fraction
actor/positive_token_count
actor/positive_token_fraction
actor/positive_mean_response_length
```

Counts used for logging must reflect real samples, not balancing padding. Aggregate them with correct global semantics.

## 7. Tests

### 7.1 Formula

On a hand-constructed batch, compare against direct token NLL for:

- one positive and one negative sequence;
- all positive;
- no positive;
- ragged response lengths;
- balancing padding;
- a threshold that changes classification.

### 7.2 Empty positive minibatch

Assert:

- loss is exactly zero;
- backward succeeds;
- no positive-LM gradient is added;
- all metrics are finite;
- PPO gradients remain unchanged.

### 7.3 Disabled equivalence

With the feature disabled, actor loss, gradients, metrics outside the new namespace, and parameters must match the Phase-2 path. Repeat with `enabled=true, coef=0`.

### 7.4 Global normalization

The same logical global minibatch partitioned across:

- one versus multiple DP ranks; and
- different microbatch sizes

must produce the same effective loss and parameter update within dtype tolerance.

### 7.5 Correctness fail-closed behavior

Test missing key, `None`, NaN, infinity, string, and vector values. Each must raise a clear error when enabled. Verify the same data trains normally when the feature is disabled.

### 7.6 Provenance

Construct a sample with positive total reward but negative configured correctness, and another with negative auxiliary reward but positive correctness. Classification must follow only the configured correctness field.

### 7.7 Repeated sampling alignment

For multiple responses sharing one prompt UID, assign mixed correctness and verify only the correct responses receive NLL loss.

## 8. Smoke test

Run v1 sync + FSDP2 + vLLM with a verifier that emits the configured correctness key:

- at least one batch with no positives;
- at least one batch with positives;
- finite PPO and positive LM losses;
- no extra actor inference forward;
- coefficient 0.1;
- actor entropy and PPO metrics still logged.

