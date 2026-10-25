"""
06_evaluate.py
==========================
"""
import os, sys, pickle, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from drl_agent import (MultiDeviceAgent, ZhaoAgent,
                        ACTIONS, COMPRESSION, OFFLOAD,
                        N_DEVICES, STATE_SIZE, N_ACT_PER)

SEED = 42; random.seed(SEED); np.random.seed(SEED)
os.makedirs("results", exist_ok=True)

# ── Load config ───────────────────────────────────────────────
cfg       = pickle.load(open("results/config.pkl", "rb"))
T_MAX     = cfg["T_MAX"];    E_MAX      = cfg["E_MAX"]
BW_MAX    = cfg["BW_MAX"];   C_EDGE_MAX = cfg["C_EDGE_MAX"]
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
ALPHA_E_  = cfg.get("ALPHA_E",   80.0)
ALPHA_B   = cfg.get("ALPHA_B",   30.0)

te_states = np.load("results/test_states.npy")
te_devs   = pickle.load(open("results/test_devices.pkl", "rb"))
y_test    = np.load("results/y_test.npy")
p_test    = np.load("results/p_test.npy")


COMP_K = {0: 8, 1: 6, 2: 4, 3: 2}   # K_0=8, K_low=6, K_med=4, K_high=2

class TaskOrientedEncoder(nn.Module):
    def __init__(self, in_dim=46, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 16),    nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(16, latent_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 2),
        )
    def forward(self, x, k=8):
        z = self.encoder(x)
        mask = torch.zeros_like(z); mask[:, :k] = 1.0
        return self.classifier(z * mask), z

ENC_CKPT    = "checkpoints/semantic_encoder.pth"
encoder     = None
enc_preds   = None   

X_test_feat = None
if os.path.exists("results/X_test.npy"):
    X_test_feat = np.load("results/X_test.npy")

if os.path.exists(ENC_CKPT) and X_test_feat is not None:
    in_dim  = X_test_feat.shape[1]
    encoder = TaskOrientedEncoder(in_dim=in_dim, latent_dim=8)
    encoder.load_state_dict(torch.load(ENC_CKPT, map_location="cpu"))
    encoder.eval()
    
    Xte_t   = torch.FloatTensor(X_test_feat)
    enc_preds = np.zeros((len(X_test_feat), 4), dtype=int)
    with torch.no_grad():
        for K_idx in range(4):
            k = COMP_K[K_idx]
            logits, _ = encoder(Xte_t, k=k)
            enc_preds[:, K_idx] = logits.argmax(1).numpy()
    print(f"[06]  compression levels.")
else:
    print("[06] WARNING:")

def load_agent(cls, ckpt):
    agent = cls()
    if os.path.exists(ckpt):
        agent.online.load_state_dict(
            torch.load(ckpt, map_location="cpu"))
        agent.online.eval()
        print(f"  ✓ Loaded {ckpt}")
    else:
        print(f"  ⚠ {ckpt} not found — using random weights")
    return agent

agent_dqn    = load_agent(MultiDeviceAgent, "checkpoints/dqn_proposed.pth")
agent_zhao   = load_agent(ZhaoAgent,        "checkpoints/zhao_dqn.pth")
agent_nopri  = load_agent(MultiDeviceAgent, "checkpoints/dqn_no_priority.pth")
agent_nosem  = load_agent(MultiDeviceAgent, "checkpoints/dqn_no_semantic.pth")
agent_fixsem = load_agent(MultiDeviceAgent, "checkpoints/dqn_fixed_semantic.pth")


def get_C_local(xi):
    return float(np.clip(75.0 - ALPHA_C * xi, 30.0, 75.0))

def get_C_edge(devs):
    avg = float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(C_EDGE_MAX - ALPHA_E_ * avg,
                         C_EDGE_MAX * 0.3, C_EDGE_MAX))

def get_BW(devs):
    avg = float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(BW_MAX - ALPHA_B * avg, 15.0, BW_MAX))

def detection_accuracy(t, O, K):
    af  = COMPRESSION[K]["acc_f"]
    xi  = t["intensity"]; pri = t["priority"]
    pb  = {2: 1.0, 1: 0.95, 0: 0.90}.get(int(pri), 0.95)
    return float(np.clip(
        (1 - O) * (0.88 - 0.50 * xi)
        + O * min(1.0, (0.76 + 0.23 * xi) * af * pb), 0, 1))

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
    return float(np.clip(
        O * (T_trans + T_comp) + T_edge + (1 - O) * T_local, 0, T_MAX))

def compute_penalty(t, O, acc, T_t, C_local):
    tau = t["tau"]
    return float(BETA_CAP * max(0.0, (1 - O) * tau - C_local)
                 + BETA_ACC * max(0.0, ALPHA_MIN - acc)
                 + BETA_LAT * max(0.0, T_t - T_MAX))

def compute_reward_device(pen, T_t, E_t, O, priority):
    w_p = W_P.get(int(priority), 0.45)
    w_t = (1 - w_p) * GAMMA_L; w_e = (1 - w_p) * GAMMA_E
    return float(np.clip(
        -(w_p * pen + w_t * T_t / T_MAX + w_e * E_t / E_MAX)
        + DELTA * O * (1 - O), -1.0, 0.1))

def allocate_bandwidth(actions):
    demands = [O * COMPRESSION[K]["ratio"] for O, K in actions]
    total   = sum(demands) + 1e-9
    return [d / total for d in demands] if total > 1.0 else demands


def policy_always_local(s, devs):  return [(0.0, 0)] * N_DEVICES
def policy_always_offload(s, devs): return [(1.0, 0)] * N_DEVICES

def policy_no_semantic(s, devs):
    with torch.no_grad():
        q = agent_nosem.online(torch.FloatTensor(s).unsqueeze(0))
    acts = []
    for qi in q:
        k0   = [j for j, (o, k) in enumerate(ACTIONS) if k == 0]
        best = k0[qi[0, k0].argmax().item()]
        acts.append(ACTIONS[best])
    return acts

def policy_no_priority(s, devs):
    with torch.no_grad():
        q = agent_nopri.online(torch.FloatTensor(s).unsqueeze(0))
    return [ACTIONS[qi.argmax().item()] for qi in q]

def policy_fixed_semantic(s, devs):
    with torch.no_grad():
        q = agent_fixsem.online(torch.FloatTensor(s).unsqueeze(0))
    acts = []
    for qi in q:
        k2   = [j for j, (o, k) in enumerate(ACTIONS) if k == 2]
        best = k2[qi[0, k2].argmax().item()]
        acts.append(ACTIONS[best])
    return acts

def policy_threshold(s, devs):
    return [(0.75 if get_C_local(t["intensity"]) < 40 else 0.0, 2)
            for t in devs]

def policy_random(s, devs):
    return [ACTIONS[random.randint(0, len(ACTIONS) - 1)]
            for _ in range(N_DEVICES)]

def policy_zhao(s, devs):
    with torch.no_grad():
        q = agent_zhao.online(
            torch.FloatTensor(agent_zhao._blind(s)).unsqueeze(0))
    return [(OFFLOAD[qi.argmax().item()], 0) for qi in q]

def policy_dqn(s, devs):
    with torch.no_grad():
        q = agent_dqn.online(torch.FloatTensor(s).unsqueeze(0))
    return [ACTIONS[qi.argmax().item()] for qi in q]

POLICIES = {
    "Always_Local":     policy_always_local,
    "Always_Offload":   policy_always_offload,
    "No_Semantic":      policy_no_semantic,
    "No_Priority":      policy_no_priority,
    "Fixed_Semantic":   policy_fixed_semantic,
    "Threshold_Based":  policy_threshold,
    "Random_Policy":    policy_random,
    "Zhao_DQN":         policy_zhao,
    "SemCIDS_Proposed": policy_dqn,
}
DISPLAY = {
    "Always_Local":     "Always Local (M1)",
    "Always_Offload":   "Always Offload (M2)",
    "No_Semantic":      "No Semantic (M3)",
    "No_Priority":      "No Priority (M4)",
    "Fixed_Semantic":   "Fixed Semantic (M5)",
    "Threshold_Based":  "Threshold-Based (M6)",
    "Random_Policy":    "Random Policy (M7)",
    "Zhao_DQN":         "Zhao et al. (M8)",
    "SemCIDS_Proposed": "DQN Proposed (M9)",
}


def evaluate(policy_fn, mkey):
    accs = []; lats = []; engs = []; bws = []; rews = []
    all_preds = []; all_true  = []

    for i in range(len(te_states)):
        s       = te_states[i]
        devs    = te_devs[i]
        C_edge  = get_C_edge(devs)
        BW_t    = get_BW(devs)
        actions = policy_fn(s, devs)   # list of (O, K) per device

        step_accs = []; step_lats = []; step_engs = []; step_rews = []
        dev_preds = []   

        for di, (O, K) in enumerate(actions):
            t       = devs[di]
            C_local = get_C_local(t["intensity"])
            acc     = detection_accuracy(t, O, K)
            T_t     = compute_latency(t, O, K, C_local, BW_t, C_edge, devs, actions)
            E_t     = compute_energy(t, O, K, C_local)
            pen     = compute_penalty(t, O, acc, T_t, C_local)
            r_i     = compute_reward_device(pen, T_t, E_t, O, t["priority"])

            step_accs.append(acc); step_lats.append(T_t)
            step_engs.append(E_t); step_rews.append(r_i)

            if enc_preds is not None:
                if O > 0:
                    flow_pred = int(enc_preds[i, K])
                else:
    
                    flow_pred = int(enc_preds[i, 0])
            else:
                flow_pred = 1 if acc > 0.5 else 0

            dev_preds.append(flow_pred)

        step_pred = 1 if sum(dev_preds) > N_DEVICES / 2 else 0
        all_preds.append(step_pred)
        all_true.append(int(y_test[i]))

        accs.append(float(np.mean(step_accs)))
        lats.append(float(np.mean(step_lats)))
        engs.append(float(np.mean(step_engs)))
        bws.append(sum(O * COMPRESSION[K]["ratio"]
                       for O, K in actions) / N_DEVICES)
        rews.append(float(np.mean(step_rews)))

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    attack_mask = all_true == 1
    benign_mask = all_true == 0
    fnr = float(((attack_mask) & (all_preds == 0)).sum() /
                (attack_mask.sum() + 1e-9))
    fpr = float(((benign_mask) & (all_preds == 1)).sum() /
                (benign_mask.sum() + 1e-9))

    return dict(
        accuracy = np.mean(accs),
        latency  = np.mean(lats),
        energy   = np.mean(engs),
        bw       = np.mean(bws),
        reward   = np.mean(rews),
        fnr      = fnr,
        fpr      = fpr,
        ep_acc   = accs,
        ep_lat   = lats,
        ep_eng   = engs,
        ep_bw    = bws,
        preds    = all_preds.tolist(),
    )


results = {}
for mkey, pfn in POLICIES.items():
    mname = DISPLAY[mkey]
    res   = evaluate(pfn, mkey)
    results[mkey] = res
   

rows_sum = [{
    "method":   DISPLAY[k],
    "accuracy": r["accuracy"],
    "latency":  r["latency"],
    "energy":   r["energy"],
    "bw":       r["bw"],
    "reward":   r["reward"],
    "fnr":      r["fnr"],
    "fpr":      r["fpr"],
} for k, r in results.items()]
pd.DataFrame(rows_sum).to_csv("results/results_summary.csv", index=False)

fnr_rows = [{
    "method":  DISPLAY[k],
    "fnr_pct": round(r["fnr"] * 100, 3),
    "fpr_pct": round(r["fpr"] * 100, 3),
} for k, r in results.items()]
pd.DataFrame(fnr_rows).to_csv("results/fnr_fpr_table.csv", index=False)
print("\n[06] FNR / FPR table (from real per-flow encoder predictions):")
for row in fnr_rows:
    print(f"  {row['method']:<28} "
          f"FNR={row['fnr_pct']:6.3f}%  FPR={row['fpr_pct']:6.3f}%")

rows_ps = []
for i in range(len(te_states)):
    row = {"sample_idx": i, "true_label": int(y_test[i]),
           "priority":   int(p_test[i])}
    for k, r in results.items():
        mname = DISPLAY[k]
        row[f"{mname}_accuracy"] = r["ep_acc"][i]
        row[f"{mname}_latency"]  = r["ep_lat"][i]
        row[f"{mname}_energy"]   = r["ep_eng"][i]
        row[f"{mname}_bw"]       = r["ep_bw"][i]
    rows_ps.append(row)
pd.DataFrame(rows_ps).to_csv("results/results_per_sample.csv", index=False)

rows_pred = []
for i in range(len(te_states)):
    row = {
        "sample_id":    i,
        "true_label":   int(y_test[i]),
        "priority":     int(p_test[i]),
        "priority_name": ({2: "High", 1: "Medium", 0: "Low"}.get(
            int(p_test[i]), "Benign") if y_test[i] == 1 else "Benign"),
    }
    for k, r in results.items():
        row[f"{k}_pred"]    = r["preds"][i]
        row[f"{k}_correct"] = int(r["preds"][i] == int(y_test[i]))
    rows_pred.append(row)
pd.DataFrame(rows_pred).to_csv("results/test_predictions_6000.csv", index=False)

pickle.dump(results, open("results/results_full.pkl", "wb"))

print("\n[06] Saved:")
print("  results/results_summary.csv      (includes fnr, fpr columns)")
print("  results/fnr_fpr_table.csv        (FNR/FPR from real per-flow predictions)")
print("  results/results_per_sample.csv")
print("  results/test_predictions_6000.csv")
print("  results/results_full.pkl")
