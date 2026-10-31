#!/usr/bin/env python3
"""Day 3 verification script — runs inside the Docker container."""
import sys
import math

# ── Step 1: Network shapes ──────────────────────────────────────────────────
sys.path.insert(0, '/workspace/catkin_ws/src/gtrl/scripts')
print("=== STEP 1: Python imports + GoT forward pass ===")
try:
    from SAC.GoalFormer import GoT
    from SAC.got_sac_network import GoTPolicy, GoTQNetwork
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Actor: GoTPolicy (Transformer + goal token) — actor_type: GaussianTransformer
    actor  = GoTPolicy(nb_actions=2, nb_pstate=2, block=2, head=4).to(device)
    # Critic: QNetwork (CNN critic, no GoT) — critic_type: CNN  ← matches config.yaml
    from SAC.got_sac_network import QNetwork
    critic = QNetwork(nb_actions=2, nb_pstate=2).to(device)

    actor_params  = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"Actor  params : {actor_params:,}")
    print(f"Critic params : {critic_params:,}")

    state  = torch.randn(4, 4, 128, 160).to(device)
    goal   = torch.randn(4, 2).to(device)
    # Actor: takes list/tuple [state, goal]
    action, log_prob, mean = actor.sample([state, goal])
    # Critic (CNN): takes list/tuple [state, goal, action]
    q1, q2 = critic([state, goal, action.detach()])

    print(f"Actor  action shape : {action.shape}")    # expect (4, 2)
    print(f"Actor log_prob shape: {log_prob.shape}")  # expect (4, 1)
    print(f"Critic Q1 shape     : {q1.shape}")        # expect (4, 2)

    assert action.shape  == (4, 2), f"FAIL: action shape {action.shape}"
    assert log_prob.shape == (4, 1), f"FAIL: log_prob shape {log_prob.shape}"
    assert q1.shape == (4, 2), f"FAIL: Q1 shape {q1.shape}"
    print(">>> STEP 1: PASS")

except Exception as e:
    print(f">>> STEP 1: FAIL — {e}")
    sys.exit(1)

# ── Step 2: TensorBoard dry-run ─────────────────────────────────────────────
print("\n=== STEP 2: TensorBoard dry-run ===")
try:
    from torch.utils.tensorboard import SummaryWriter
    log_dir = '/workspace/catkin_ws/src/gtrl/scripts/SAC/runs/dry_run_day3'
    writer = SummaryWriter(log_dir)
    for i in range(100):
        writer.add_scalar('Test/reward_episode', math.sin(i / 10) * 50, i)
        writer.add_scalar('Test/reward_mean20',  math.sin(i / 10) * 40, i)
        writer.add_scalar('SAC/temperature',     0.5 - i * 0.003,      i)
    writer.close()
    print(f"TensorBoard events written to: {log_dir}")
    print(">>> STEP 2: PASS")

except Exception as e:
    print(f">>> STEP 2: FAIL — {e}")
    sys.exit(1)

print("\n=== ALL DAY 3 CHECKS PASSED ===")
