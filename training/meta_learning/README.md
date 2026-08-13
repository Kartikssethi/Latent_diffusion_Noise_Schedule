# Meta-Learning Integration Guide (Small, Practical)

This folder is a placeholder scaffold for adding meta-learning to adaptive diffusion noise scheduling.

## Goal

Use training feedback (loss trends, sample quality signals, stability metrics) to let a meta-learner adjust the diffusion noise schedule during training.

## Why This Exists

Instead of keeping one fixed noise schedule for all training stages, we want a controller that adapts the schedule based on current model behavior.

## Components and Responsibilities

- `training/feedback/training_feedback.py`
  - Collect raw feedback from training steps/epochs.
  - Normalize feedback into a consistent structure.

- `training/feedback/reward_signals.py`
  - Convert feedback into reward/quality signals for meta-learning.

- `training/meta_learning/meta_update_loop.py`
  - Define when and how meta-learner updates happen.
  - Trigger policy/state updates from reward signals.

- `training/meta_learning/scheduler_bridge.py`
  - Bridge meta-learner output to scheduler input.
  - Validate and apply schedule deltas safely.

- `scheduler/adaptive_noise_scheduler.py`
  - Owns current schedule state and update interface.

- `scheduler/schedule_state.py`
  - Stores the current schedule and history/checkpoints.

## Incorporation Flow

1. Train diffusion model for a short interval (step window or mini-epoch).
2. Collect training feedback.
3. Compute reward/health signals.
4. Run meta-update loop to propose schedule adjustments.
5. Pass proposed adjustment through scheduler bridge.
6. Adaptive scheduler applies bounded update.
7. Continue training with the new schedule.
8. Repeat.

## Safety Rules for Early Versions

- Clip updates to a small delta per interval.
- Keep fallback to previous stable schedule.
- Log every schedule change with timestamp and reason.
- Reject invalid updates (NaN, exploding variance, out-of-range betas).

## Suggested Minimal Milestones

1. Feedback schema finalized.
2. Reward signal function implemented.
3. Static bridge with manual mock schedule updates.
4. Meta-update loop with no-learning baseline policy.
5. End-to-end dry run with logs only.
6. Real meta-learner training enabled.

## Notes

This is intentionally documentation-only and placeholder-oriented.
Implementation should begin by defining data contracts between feedback, reward computation, and scheduler bridge.
