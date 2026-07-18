# VAPO

VAPO extends synchronous actor-critic PPO with independently targeted actor and
critic GAE, a response-length-dependent actor lambda, critic-only value warmup,
asymmetric PPO clipping, and an auxiliary language-model loss on
verifier-confirmed correct trajectories. Repeated sampling is controlled by the
existing `actor_rollout_ref.rollout.n`; it does not introduce group-relative
normalization.

The initial implementation supports only the v1 synchronous trainer, FSDP2
actor/reference/critic, vLLM rollout, and text-only single-turn trajectories.
Enabling any VAPO component on another stack fails at startup.

VAPO does not require a format bonus or another shaped reward. The paper
primarily omits KL loss; the canonical recipe therefore disables both
`algorithm.use_kl_in_reward` and `actor_rollout_ref.actor.use_kl_loss`. The
general implementation still composes with verl's existing optional KL
configuration, but positive-example selection always uses the configured raw
verifier output rather than a KL-shaped reward.

## Configuration

- `algorithm.gae.decoupled` computes actor advantages and critic returns
  independently.
- `algorithm.gae.policy_lambda` is the scalar actor lambda when length
  adaptation is disabled.
- `algorithm.gae.critic_lambda` controls critic targets.
- `algorithm.gae.length_adaptive` enables
  `1 - 1 / (length_alpha * observed_response_length)`.
- `algorithm.gae.length_alpha` defaults to `0.05`.
- `algorithm.gae.policy_lambda_min` and `policy_lambda_max` clamp the formula to
  `[0, 1]` by default. With the paper alpha and lower bound, lengths up to 20
  clamp to zero; this edge behavior is an implementation decision because the
  paper does not specify it.
- `actor_rollout_ref.actor.positive_lm.*` selects a numeric scalar from reward
  extra information, thresholds it, and adds token NLL for positive responses.
  The reserved key `rm_scores` explicitly selects the raw terminal verifier
  score before KL-in-reward or other shaping. Missing or non-finite correctness
  values fail the batch before actor update.
- `trainer.value_warmup_steps` counts outer rollout/update iterations. During
  these iterations the critic updates and the actor optimizer does not step.
- `trainer.save_value_warmup_checkpoint` saves a normal resumable checkpoint
  after the final warmup critic update. Resume it to continue the same run, or
  initialize a new run's critic from it with `value_warmup_steps=0`.

Run the small canonical recipe with:

```bash
bash examples/vapo_trainer/run_qwen3_8b_fsdp2.sh
```

Run an ablation while holding generated trajectory count fixed with:

```bash
TOTAL_TRAJECTORIES=512 bash examples/vapo_trainer/run_ablation.sh full_n16
```

Available names are `vanilla_ppo`, `token_mean_clip_higher`,
`decoupled_gae`, `length_adaptive_gae`, `value_pretraining`,
`positive_lm`, `full_n16`, and `full_n1`.

The paper-scale allocation uses 512 unique prompts, 16 responses per prompt
(8192 trajectories), actor and critic minibatches of 512, actor/critic learning
rates of `1e-6`/`2e-6`, 50 warmup iterations, gamma 1, critic lambda 1, length
alpha 0.05, clipping 0.20/0.28, and positive-LM coefficient 0.1. The checked-in
recipe reduces prompt and minibatch counts and is a smoke recipe, not a claim of
reproducing the paper's reported score.
