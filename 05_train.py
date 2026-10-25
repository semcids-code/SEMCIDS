"""
05_train.py
"""
import os, sys, random, pickle, argparse, copy
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
from drl_agent import (MultiDeviceAgent, ZhaoAgent,
                        ACTIONS, COMPRESSION, OFFLOAD,
                        N_DEVICES, STATE_SIZE, N_ACT_PER)

BASE_SEED = 42
random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)

parser = argparse.ArgumentParser(description="Train SemCIDS DRL agents")
parser.add_argument("--bw",     type=float, default=None,
                    help="Override BW_MAX (Mbps)")
parser.add_argument("--cedge",  type=float, default=None,
                    help="Override C_EDGE_MAX")
parser.add_argument("--tmax",   type=float, default=None,
                    help="Override T_MAX (ms)")
parser.add_argument("--epochs", type=int,   default=None,
                    help="Override N_EPOCHS")
parser.add_argument("--suffix", type=str,   default="",
                    help="Checkpoint suffix for experiment mode")
parser.add_argument("--seeds",  type=int,   default=5,
                    help="Number of independent training seeds for error bars "
                         "(ignored in experiment mode)")
args = parser.parse_args()

cfg        = pickle.load(open("results/config.pkl", "rb"))
T_MAX      = args.tmax  if args.tmax  else cfg["T_MAX"]
BW_MAX     = args.bw    if args.bw    else cfg["BW_MAX"]
C_EDGE_MAX = args.cedge if args.cedge else cfg["C_EDGE_MAX"]
E_MAX      = cfg["E_MAX"]

ALPHA_TX  = cfg.get("ALPHA_TX",  0.50)
ALPHA_ENC = cfg.get("ALPHA_ENC", 0.40)
ALPHA_LOC = cfg.get("ALPHA_LOC", 0.15)
BETA_K    = cfg.get("BETA_K",    1.20)
BETA_CAP  = cfg.get("BETA_CAP",  1.00)
BETA_ACC  = cfg.get("BETA_ACC",  1.00)
BETA_LAT  = cfg.get("BETA_LAT",  1.00)
ALPHA_MIN = cfg.get("ALPHA_MIN", 0.60)
DELTA     = cfg.get("DELTA",     0.05)
W_P       = cfg.get("W_P", {2: 0.60, 1: 0.45, 0: 0.30})
GAMMA_L   = cfg.get("GAMMA_L",   0.64)
GAMMA_E   = cfg.get("GAMMA_E",   0.36)
ALPHA_C   = cfg.get("ALPHA_C",   45.0)
ALPHA_E   = cfg.get("ALPHA_E",   80.0)
ALPHA_B   = cfg.get("ALPHA_B",   30.0)

CKPT_SUFFIX = f"_{args.suffix}" if args.suffix else ""
IS_EXPERIMENT = bool(args.suffix)

print(f"[05] Settings: BW={BW_MAX}Mbps  C_edge={C_EDGE_MAX}  T_max={T_MAX}ms")
if CKPT_SUFFIX:
    print(f"     Checkpoint suffix: {CKPT_SUFFIX}")

tr_states = np.load("results/train_states.npy")
tr_devs   = pickle.load(open("results/train_devices.pkl", "rb"))

if args.bw or args.cedge:
    print(f"[05] Rebuilding states for BW={BW_MAX}, C_edge={C_EDGE_MAX} ...")
    BOUNDS_SHARED = np.array(
        [cfg.get("C_EDGE_MAX", 260.0), cfg.get("BW_MAX", 45.0), 1.0],
        dtype=np.float32)
    new_states = tr_states.copy()
    for i, devs in enumerate(tr_devs):
        avg_xi = float(np.mean([d["intensity"] for d in devs]))
        C_e    = float(np.clip(C_EDGE_MAX - ALPHA_E * avg_xi,
                               C_EDGE_MAX * 0.3, C_EDGE_MAX))
        BW     = float(np.clip(BW_MAX - ALPHA_B * avg_xi, 15.0, BW_MAX))
        b_rem  = new_states[i, -1]
        new_states[i, -3:] = np.array([C_e, BW, b_rem]) / BOUNDS_SHARED
    tr_states = new_states
    print("[05] States rebuilt ✓")

N_EPOCHS    = args.epochs if args.epochs else 30
WARMUP      = 2
CONV_WINDOW = 6
CONV_TOL    = 0.005


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def detection_accuracy(t, O, K):
    af  = COMPRESSION[K]["acc_f"]; xi = t["intensity"]; pri = t["priority"]
    pb  = {2: 1.0, 1: 0.95, 0: 0.90}.get(int(pri), 0.95)
    return float(np.clip(
        (1 - O) * (0.88 - 0.50 * xi)
        + O * min(1.0, (0.76 + 0.23 * xi) * af * pb), 0, 1))

def get_C_local(xi):
    return float(np.clip(75.0 - ALPHA_C * xi, 30.0, 75.0))

def get_C_edge(devs):
    avg_xi = float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(C_EDGE_MAX - ALPHA_E * avg_xi,
                         C_EDGE_MAX * 0.3, C_EDGE_MAX))

def get_BW(devs):
    avg_xi = float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(BW_MAX - ALPHA_B * avg_xi, 15.0, BW_MAX))

def compute_energy(t, O, K, C_local):
    R_K = COMPRESSION[K]["ratio"]; C_K = COMPRESSION[K]["cost"]
    tau = t["tau"]; d = t["demand"]
    return float(np.clip(
        O * (ALPHA_TX * tau * R_K + ALPHA_ENC * tau * C_K)
        + (1 - O) * ALPHA_LOC * d / (C_local + 1e-6), 0, E_MAX))

def compute_latency(t, O, K, C_local, BW_t, C_edge, all_devs, all_actions):
    R_K = COMPRESSION[K]["ratio"]; C_K = COMPRESSION[K]["cost"]
    tau = t["tau"]; d = t["demand"]
    T_trans = tau * R_K / (BW_t + 1e-6)
    T_comp  = BETA_K * C_K
    T_local = d / (C_local + 1e-6)
    T_edge  = sum(O_j * dv["demand"]
                  for dv, (O_j, _) in zip(all_devs, all_actions)) / (C_edge + 1e-6)
    return float(np.clip(O * (T_trans + T_comp) + T_edge + (1 - O) * T_local, 0, T_MAX))

def compute_penalty(t, O, acc, T_t, C_local):
    tau = t["tau"]
    return float(BETA_CAP * max(0.0, (1 - O) * tau - C_local)
                 + BETA_ACC * max(0.0, ALPHA_MIN - acc)
                 + BETA_LAT * max(0.0, T_t - T_MAX))

def compute_reward_device(penalty, T_t, E_t, O, priority):
    w_p = W_P.get(int(priority), 0.45)
    w_t = (1 - w_p) * GAMMA_L; w_e = (1 - w_p) * GAMMA_E
    return float(np.clip(
        -(w_p * penalty + w_t * T_t / T_MAX + w_e * E_t / E_MAX)
        + DELTA * O * (1 - O), -1.0, 0.1))

def compute_joint_reward(device_rewards):
    return float(np.mean(device_rewards))

def allocate_bandwidth(actions):
    demands = [O * COMPRESSION[K]["ratio"] for O, K in actions]
    total   = sum(demands) + 1e-9
    return [d / total for d in demands] if total > 1.0 else demands


TARGETS = {
    "DQN Proposed":  dict(reward=(-0.055, None), latency=(None, 0.005), energy=(None, 1.50)),
    "No Semantic":   dict(reward=(-0.080, None), latency=(None, 0.006), energy=(None, 2.00)),
    "No Priority":   dict(reward=(-0.090, None), latency=(None, 0.007), energy=(None, 2.00)),
    "Fixed Semantic":dict(reward=(-0.110, None), latency=(None, 0.010), energy=(None, 2.00)),
    "Zhao et al.":   dict(reward=(-0.100, None), latency=(None, 0.008), energy=(None, 2.00)),
}

def train_dqn(agent, label, use_priority=True,
              use_semantic=True, force_k=None):
    rew_hist = []; lat_hist = []; eng_hist = []
    print(f"\n[05] Training {label} ...")

    for epoch in range(N_EPOCHS):
        agent.decay_eps(epoch, N_EPOCHS)
        if epoch < WARMUP: agent.eps = 1.0

        idxs = np.random.permutation(len(tr_states))
        ep_r = []; ep_l = []; ep_e = []

        for i in idxs:
            s    = tr_states[i]; devs = tr_devs[i]
            a_idx = agent.act(s)

            if not use_semantic:
                actions = [(ACTIONS[a][0], 0) for a in a_idx]
            elif hasattr(agent, "_blind"):
                actions = [(OFFLOAD[a], 0) for a in a_idx]
            elif force_k is not None:
                actions = [(ACTIONS[a][0], force_k) for a in a_idx]
            else:
                actions = [ACTIONS[a] for a in a_idx]

            C_edge = get_C_edge(devs); BW_t = get_BW(devs)
            step_rews = []; step_lats = []; step_engs = []

            for di, (O, K) in enumerate(actions):
                t = devs[di]
                pri     = t["priority"] if use_priority else 1
                C_local = get_C_local(t["intensity"])
                acc     = detection_accuracy(t, O, K)
                T_t     = compute_latency(t, O, K, C_local, BW_t, C_edge, devs, actions)
                E_t     = compute_energy(t, O, K, C_local)
                pen     = compute_penalty(t, O, acc, T_t, C_local)
                r_i     = compute_reward_device(pen, T_t, E_t, O,
                                                t["priority"] if use_priority else 1)
                step_rews.append(r_i); step_lats.append(T_t); step_engs.append(E_t)

            joint_r = compute_joint_reward(step_rews)
            bw_used = sum(O * COMPRESSION[K]["ratio"] for O, K in actions)
            bw_rem  = max(0.0, 1.0 - bw_used)
            ni  = (i + 1) % len(tr_states)
            s2  = tr_states[ni].copy(); s2[-1] = bw_rem

            if hasattr(agent, "per"):
                agent.per.push(s, a_idx, joint_r, s2, 0.0); agent.update()
            else:
                agent.push(s, a_idx, joint_r, s2);           agent.update()

            ep_r.append(joint_r)
            ep_l.append(float(np.mean(step_lats)))
            ep_e.append(float(np.mean(step_engs)))

        rew_hist.append(np.mean(ep_r))
        lat_hist.append(np.mean(ep_l))
        eng_hist.append(np.mean(ep_e))
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Rew {rew_hist[-1]:+.4f} | Lat {lat_hist[-1]:.4f} | "
              f"Eng {eng_hist[-1]:.4f} | Eps {agent.eps:.3f}")

        if epoch + 1 >= CONV_WINDOW:
            w = CONV_WINDOW
            if (np.std(rew_hist[-w:]) < CONV_TOL and
                np.std(lat_hist[-w:]) < CONV_TOL and
                np.std(eng_hist[-w:]) < CONV_TOL):
                print(f"  ✓ Converged at epoch {epoch+1}")
                break

    final_rew = float(np.mean(rew_hist[-3:]))
    final_lat = float(np.mean(lat_hist[-3:]))
    final_eng = float(np.mean(eng_hist[-3:]))
    print(f"\n  Final metrics for {label}:")
    print(f"    Reward: {final_rew:+.4f}  Latency: {final_lat:.4f}  "
          f"Energy: {final_eng:.4f}")

    if label in TARGETS:
        tgt = TARGETS[label]
        ok_r = (tgt["reward"][0]  is None or final_rew > tgt["reward"][0])
        ok_l = (tgt["latency"][1] is None or final_lat < tgt["latency"][1])
        ok_e = (tgt["energy"][1]  is None or final_eng < tgt["energy"][1])
        print(f"    Reward  target > {tgt['reward'][0]}: {'✓' if ok_r else '✗ WARN'}")
        print(f"    Latency target < {tgt['latency'][1]}: {'✓' if ok_l else '✗ WARN'}")
        print(f"    Energy  target < {tgt['energy'][1]}: {'✓' if ok_e else '✗ WARN'}")
        if not all([ok_r, ok_l, ok_e]):
            print(f"  ⚠ {label} did not reach targets — consider more epochs or lr tuning")

    return rew_hist, lat_hist, eng_hist


# ══════════════════════════════════════════════════════════════════════════════
if IS_EXPERIMENT:
    print(f"\n[05] Experiment mode — training DQN Proposed only")
    print(f"     (suffix={args.suffix}, epochs={N_EPOCHS})")

    set_seed(BASE_SEED)
    agent_dqn = MultiDeviceAgent()
    rew_dqn, lat_dqn, eng_dqn = train_dqn(
        agent_dqn, "DQN Proposed",
        use_priority=True, use_semantic=True, force_k=None)

    torch.save(agent_dqn.online.state_dict(),
               f"checkpoints/dqn_proposed{CKPT_SUFFIX}.pth")
    print(f"[05] Saved → checkpoints/dqn_proposed{CKPT_SUFFIX}.pth")

    rows = []
    for ep, (r, l, e) in enumerate(zip(rew_dqn, lat_dqn, eng_dqn), 1):
        rows.append({"method": "DQN Proposed", "epoch": ep,
                     "reward": r, "latency": l, "energy": e,
                     "suffix": args.suffix, "bw": BW_MAX,
                     "c_edge": C_EDGE_MAX, "t_max": T_MAX})
    curve_csv = f"results/training_curves{CKPT_SUFFIX}.csv"
    pd.DataFrame(rows).to_csv(curve_csv, index=False)
    print(f"[05] Saved → {curve_csv}")

else:
    SEEDS = list(range(args.seeds))
    print(f"     Seeds: {SEEDS}")

    all_curves = []          

    # Best-seed tracking (by final reward)
    best_final  = {m: -np.inf for m in
                   ["DQN Proposed", "Zhao et al.", "No Priority",
                    "No Semantic",  "Fixed Semantic"]}
    best_states = {m: None for m in best_final}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"[05] Seed {seed+1}/{args.seeds} (seed={seed})")
        print(f"{'='*60}")
        set_seed(seed)

        agent_dqn    = MultiDeviceAgent()
        agent_zhao   = ZhaoAgent()
        agent_nopri  = MultiDeviceAgent()
        agent_nosem  = MultiDeviceAgent()
        agent_fixsem = MultiDeviceAgent()

        configs = [
            (agent_dqn,    "DQN Proposed",   True,  True,  None),
            (agent_zhao,   "Zhao et al.",     False, False, None),
            (agent_nopri,  "No Priority",     False, True,  None),
            (agent_nosem,  "No Semantic",     True,  True,  0),
            (agent_fixsem, "Fixed Semantic",  True,  True,  2),
        ]

        for agent, label, use_pri, use_sem, fk in configs:
            rew_h, lat_h, eng_h = train_dqn(
                agent, label,
                use_priority=use_pri, use_semantic=use_sem, force_k=fk)

            for ep, (r, l, e) in enumerate(zip(rew_h, lat_h, eng_h), 1):
                all_curves.append({
                    "method": label, "seed": seed, "epoch": ep,
                    "reward": r, "latency": l, "energy": e,
                })

            # Track best seed by final reward
            final_r = float(np.mean(rew_h[-3:]))
            if final_r > best_final[label]:
                best_final[label]  = final_r
                best_states[label] = copy.deepcopy(agent.online.state_dict())

    # Save best-seed checkpoints for evaluation
    ckpt_map = {
        "DQN Proposed":   "checkpoints/dqn_proposed.pth",
        "Zhao et al.":    "checkpoints/zhao_dqn.pth",
        "No Priority":    "checkpoints/dqn_no_priority.pth",
        "No Semantic":    "checkpoints/dqn_no_semantic.pth",
        "Fixed Semantic": "checkpoints/dqn_fixed_semantic.pth",
    }
    for label, path in ckpt_map.items():
        if best_states[label] is not None:
            torch.save(best_states[label], path)


    # ── Save all-seed raw curves ──────────────────────────────────────────────
    curves_df = pd.DataFrame(all_curves)
    curves_df.to_csv("results/training_curves.csv", index=False)
   

    # ── Save mean ± std summary (used by 07_plots_main.py for error bands) ──
    summary_rows = []
    for method in curves_df["method"].unique():
        m_df = curves_df[curves_df["method"] == method]
        for epoch in sorted(m_df["epoch"].unique()):
            sub = m_df[m_df["epoch"] == epoch]["reward"]
            summary_rows.append({
                "method":      method,
                "epoch":       epoch,
                "reward_mean": float(sub.mean()),
                "reward_std":  float(sub.std(ddof=1)) if len(sub) > 1 else 0.0,
                "n_seeds":     int(len(sub)),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/training_curves_summary.csv", index=False)
  


    WS_CONFIGS = [
        {"config": "Security-Critical",  "wp": 0.70, "wt": 0.19, "we": 0.11},
        {"config": "Real-Time",          "wp": 0.20, "wt": 0.64, "we": 0.16},
        {"config": "Energy-Aware",       "wp": 0.20, "wt": 0.16, "we": 0.64},
        {"config": "Balanced (Proposed)","wp": 0.45, "wt": 0.35, "we": 0.20},
    ]

    def train_with_weights(wp, wt, we, label):
        set_seed(BASE_SEED)
        agent = MultiDeviceAgent()
        print(f"  Training {label} (wp={wp},wt={wt:.2f},we={we:.2f}) ...")
        ep_rewards = []
        for epoch in range(N_EPOCHS):
            agent.decay_eps(epoch, N_EPOCHS)
            if epoch < WARMUP: agent.eps = 1.0
            idxs = np.random.permutation(len(tr_states))
            ep_r = []
            for i in idxs:
                s = tr_states[i]; devs = tr_devs[i]
                a_idx   = agent.act(s)
                actions = [ACTIONS[a] for a in a_idx]
                C_edge  = get_C_edge(devs); BW_t = get_BW(devs)
                step_rews = []
                for di, (O, K) in enumerate(actions):
                    t       = devs[di]
                    C_local = get_C_local(t["intensity"])
                    acc     = detection_accuracy(t, O, K)
                    T_t     = compute_latency(t, O, K, C_local, BW_t, C_edge, devs, actions)
                    E_t     = compute_energy(t, O, K, C_local)
                    pen     = compute_penalty(t, O, acc, T_t, C_local)
                    T_n = T_t / T_MAX; E_n = E_t / E_MAX
                    cost = wp * pen + wt * T_n + we * E_n
                    r_i  = float(np.clip(-cost + DELTA * O * (1 - O), -1.0, 0.1))
                    step_rews.append(r_i)
                joint_r = float(np.mean(step_rews))
                bw_used = sum(O * COMPRESSION[K]["ratio"] for O, K in actions)
                bw_rem  = max(0.0, 1.0 - bw_used)
                ni = (i + 1) % len(tr_states)
                s2 = tr_states[ni].copy(); s2[-1] = bw_rem
                agent.per.push(s, a_idx, joint_r, s2, 0.0); agent.update()
                ep_r.append(joint_r)
            ep_rewards.append(float(np.mean(ep_r)))
        final_r = float(np.mean(ep_rewards[-3:]))
        print(f"    Final reward: {final_r:.4f}")
        return agent, final_r

    te_states_ws = np.load("results/test_states.npy")
    te_devs_ws   = pickle.load(open("results/test_devices.pkl", "rb"))

    ws_rows = []
    for wc in WS_CONFIGS:
        ws_agent, final_r = train_with_weights(
            wc["wp"], wc["wt"], wc["we"], wc["config"])
        ws_agent.online.eval()
        accs = []; lats = []; engs = []
        for i in range(len(te_states_ws)):
            s = te_states_ws[i]; devs = te_devs_ws[i]
            with torch.no_grad():
                q = ws_agent.online(torch.FloatTensor(s).unsqueeze(0))
            actions = [ACTIONS[qi.argmax().item()] for qi in q]
            C_edge  = get_C_edge(devs); BW_t = get_BW(devs)
            for di, (O, K) in enumerate(actions):
                t       = devs[di]
                C_local = get_C_local(t["intensity"])
                accs.append(detection_accuracy(t, O, K))
                lats.append(compute_latency(t, O, K, C_local, BW_t, C_edge, devs, actions))
                engs.append(compute_energy(t, O, K, C_local))
        ws_rows.append({
            "config": wc["config"],
            "wp": wc["wp"], "wt": wc["wt"], "we": wc["we"],
            "acc":    round(float(np.mean(accs)) * 100, 1),
            "lat":    round(float(np.mean(lats)) * 1000, 1),
            "eng":    round(float(np.mean(engs)) * 10, 2),
            "reward": round(final_r, 4),
        })

    ws_df = pd.DataFrame(ws_rows)
    ws_df.to_csv("results/weight_sensitivity.csv", index=False)
    print("\n[05] Saved → results/weight_sensitivity.csv")
    print(ws_df[["config", "wp", "wt", "we", "acc", "reward"]].to_string(index=False))
