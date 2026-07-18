# Phase 4: VAPO Recipe, End-to-End Validation, and Handoff

**Prerequisite:** Phases 1–3 landed and passed acceptance tests.  
**Scope:** supported-stack validation, example recipe, ablations, documentation, and final quality gates.

## 1. Supported recipe

Provide one canonical text-only recipe using:

```text
v1 PPO trainer
synchronous trainer mode
FSDP2 actor/reference/critic
vLLM rollout
GAE actor-critic optimization
verifier reward with an explicit correctness field
```

The recipe should expose environment-variable overrides for model path, data paths, device count, batch sizes, maximum lengths, and output directory while pinning algorithm defaults.

Required VAPO settings:

```yaml
algorithm:
  adv_estimator: gae
  gamma: 1.0
  gae:
    decoupled: true
    critic_lambda: 1.0
    length_adaptive: true
    length_alpha: 0.05
    policy_lambda_min: 0.0
    policy_lambda_max: 1.0

actor_rollout_ref:
  actor:
    clip_ratio_low: 0.20
    clip_ratio_high: 0.28
    loss_agg_mode: token-mean
    positive_lm:
      enabled: true
      coef: 0.1
      correctness_key: acc
      correctness_threshold: 1.0
  rollout:
    n: 16

trainer:
  value_warmup_steps: 50
```

Use the current repository's exact configuration paths and generated-config workflow. Do not manually edit generated configuration files if repository tooling generates them.

## 2. Paper-scale values versus runnable defaults

Document the paper values:

- 512 unique prompts per sampling iteration;
- 16 responses per prompt;
- 8192 generated trajectories;
- actor and critic minibatch size 512;
- actor learning rate `1e-6`;
- critic learning rate `2e-6`;
- warmup-constant scheduler;
- 50 value-warmup steps;
- `gamma=1`;
- critic lambda 1;
- length alpha 0.05;
- clip low/high 0.20/0.28;
- positive LM coefficient 0.1.

The checked-in smoke recipe may use smaller batch/model settings so it can run on accessible hardware. Label every deviation from paper scale. Do not represent the small recipe as a reproduction of the reported result.

## 3. Startup validation

When the canonical VAPO configuration is enabled, fail clearly unless:

- `trainer.use_v1=true`;
- `trainer.v1.trainer_mode=sync`;
- actor/reference/critic use FSDP2;
- rollout backend is vLLM;
- a critic is enabled;
- advantage estimator is GAE;
- inputs are supported text-only single-turn trajectories;
- the configured correctness field is available when positive LM loss is enabled.

Where correctness-field availability cannot be known until the first rewarded batch, validate it at that boundary before any actor update.

## 4. End-to-end tests

### 4.1 Disabled baseline

Run the post-change code with every new feature disabled. Under deterministic settings, compare against the pinned pre-change v1 PPO baseline:

- batch fields;
- actor advantages;
- critic returns;
- actor and critic losses;
- parameter checksums after multiple steps.

Document any nondeterminism that prevents bitwise comparison and use the tightest justified numerical comparison.

### 4.2 Full small-model run

Run long enough to cross:

- all 50 value-warmup steps;
- the first actor update;
- multiple actor and critic PPO updates;
- batches with and without correct samples.

Require:

- no NaN/Inf;
- actor unchanged during warmup;
- critic updating during warmup;
- actor updating after warmup;
- finite explained variance;
- length-adaptive lambda metrics;
- asymmetric clipping metrics;
- positive LM metrics;
- repeated prompt samples correctly represented;
- checkpoints save and load.

### 4.3 Resume

Checkpoint:

- during value warmup;
- exactly at warmup completion; and
- during joint training.

Resume each and verify correct warmup/actor-update behavior without duplicated or skipped steps.

### 4.4 Unsupported stack

Test clear startup rejection for async trainer mode and at least one unsupported training engine. Do not allow a run to begin and fail much later inside a worker.

## 5. Algorithmic ablations

Provide scripts or configuration overlays for:

1. vanilla v1 PPO;
2. token mean + Clip-Higher only;
3. decoupled GAE;
4. decoupled + length-adaptive GAE;
5. previous + value pretraining;
6. previous + positive LM loss;
7. full VAPO with `rollout.n=16`;
8. full VAPO with `rollout.n=1`.

For sampling ablations, hold total generated trajectories or total generated response tokens approximately fixed and report which constraint was used. Changing `rollout.n` without compensating unique prompt count is a compute and batch-size change, not a clean group-sampling ablation.

Every run must record the resolved configuration, code revision, model/data identifiers, random seed, total generated tokens, and wall-clock time.

## 6. Monitoring

Dashboards should put the following on one view:

- train/eval verifier accuracy;
- response-length mean, p95, and maximum-length fraction;
- actor entropy;
- actor KL and clip fractions;
- policy-lambda distribution;
- critic value loss and explained variance;
- actor and critic learning rates;
- positive-example frequency and LM loss;
- throughput and generated-token count.

Value collapse warning signs include falling explained variance, extreme value magnitude, response-length collapse, and degradation immediately after actor updates begin. Log and diagnose; do not add automatic intervention in the initial implementation.

## 7. Documentation

Add an algorithm document that:

- describes each VAPO modification relative to PPO;
- distinguishes already-general verl primitives from new implementation;
- states the supported execution stack;
- documents every new configuration field;
- gives the canonical command;
- explains value-warmup checkpoint reuse;
- explains the short-sequence lambda clamp;
- identifies the correctness-field requirement;
- labels public-paper details versus implementation decisions;
- avoids claiming exact paper reproduction.

## 8. Final quality gate

Before considering implementation complete:

- run focused CPU unit tests for GAE, masking, metrics, and positive LM loss;
- run FSDP2 distributed tests for loss normalization;
- run the v1 synchronous end-to-end smoke test;
- run formatting, lint, and relevant pre-commit hooks;
- review every generated configuration change;
- confirm no legacy v0-only trainer path was accidentally modified to carry unsupported semantics;
- confirm no unrelated working-tree changes are included.

If preparing a contribution, follow the repository's duplicate-work checks and AI-assistance disclosure requirements before opening a pull request.

