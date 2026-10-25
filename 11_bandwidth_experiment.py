"""
11_bandwidth_experiment.py
=================================
"""
import os, sys, subprocess, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from drl_agent import (MultiDeviceAgent,
                        ACTIONS, COMPRESSION,
                        N_DEVICES)

os.makedirs("results",     exist_ok=True)
os.makedirs("Images",      exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

cfg       = pickle.load(open("results/config.pkl","rb"))
N         = cfg["N_DEVICES"]
ALPHA_C   = cfg.get("ALPHA_C",   45.0)
ALPHA_E_  = cfg.get("ALPHA_E",   80.0)
ALPHA_B   = cfg.get("ALPHA_B",   30.0)
ALPHA_TX  = cfg.get("ALPHA_TX",  0.50)
ALPHA_ENC = cfg.get("ALPHA_ENC", 0.40)
ALPHA_LOC = cfg.get("ALPHA_LOC", 0.15)
BETA_K    = cfg.get("BETA_K",    1.20)

tr_states_base = np.load("results/train_states.npy")
te_states_base = np.load("results/test_states.npy")
tr_devs   = pickle.load(open("results/train_devices.pkl","rb"))
te_devs   = pickle.load(open("results/test_devices.pkl","rb"))

BOUNDS_SHARED = np.array(
    [cfg.get("C_EDGE_MAX",260.0),
     cfg.get("BW_MAX",45.0),
     1.0], dtype=np.float32)

# ── Helpers ───────────────────────────────────────────────────
def get_C_local(xi):
    return float(np.clip(75.0-ALPHA_C*xi,30.0,75.0))

def get_C_edge_val(devs, C_edge_max):
    avg=float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(
        C_edge_max-ALPHA_E_*avg,
        C_edge_max*0.3, C_edge_max))

def get_BW_val(devs, BW_max):
    avg=float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(
        BW_max-ALPHA_B*avg, 15.0, BW_max))

def detection_accuracy(t, O, K):
    af=COMPRESSION[K]["acc_f"]; xi=t["intensity"]
    pri=t["priority"]
    pb={2:1.0,1:0.95,0:0.90}.get(int(pri),0.95)
    return float(np.clip(
        (1-O)*(0.88-0.50*xi)
        +O*min(1.0,(0.76+0.23*xi)*af*pb),0,1))

def compute_latency_simple(t, O, K, C_local,
                            BW_t, C_edge,
                            all_devs, all_actions,
                            T_max):
    R_K=COMPRESSION[K]["ratio"]; C_K=COMPRESSION[K]["cost"]
    tau=t["tau"]; d=t["demand"]
    T_trans=tau*R_K/(BW_t+1e-6)
    T_comp =BETA_K*C_K
    T_local=d/(C_local+1e-6)
    T_edge =sum(O_j*dv["demand"]
        for dv,(O_j,_) in zip(all_devs,all_actions)
    )/(C_edge+1e-6)
    T=O*(T_trans+T_comp)+T_edge+(1-O)*T_local
    return float(np.clip(T,0,T_max))

def rebuild_states(base_states, devs_list,
                   BW_max, C_edge_max):
    """Update shared dims [C_edge,BW,b_rem] in states."""
    new = base_states.copy()
    for i,devs in enumerate(devs_list):
        avg=float(np.mean(
            [d["intensity"] for d in devs]))
        C_e=float(np.clip(
            C_edge_max-ALPHA_E_*avg,
            C_edge_max*0.3, C_edge_max))
        BW =float(np.clip(
            BW_max-ALPHA_B*avg, 15.0, BW_max))
        b_rem=new[i,-1]
        shared=np.array([C_e,BW,b_rem],
                         dtype=np.float32)
        new[i,-3:]=shared/BOUNDS_SHARED
    return new

# ── Step 1: Train using 05_train.py ──────────────────────────
def train_setting(BW_max, C_edge_max, T_max,
                  suffix, n_epochs=15):
    """
    Call 05_train.py with given settings.
    Saves checkpoint to checkpoints/dqn_proposed_{suffix}.pth
    """
    ckpt=f"checkpoints/dqn_proposed_{suffix}.pth"
    if os.path.exists(ckpt):
        print(f"  [skip] {ckpt} already exists")
        return ckpt

    cmd = [
        sys.executable, "05_train.py",
        "--bw",     str(BW_max),
        "--cedge",  str(C_edge_max),
        "--tmax",   str(T_max),
        "--epochs", str(n_epochs),
        "--suffix", suffix,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠ Training failed for {suffix}")
        return None
    return ckpt

# ── Step 2: Evaluate at given setting ────────────────────────
def evaluate_setting(ckpt_path, BW_max,
                     C_edge_max, T_max):
    """
    Load trained checkpoint and evaluate
    SemCIDS vs Fixed-Semantic vs Always-Offload.
    """
    te_st = rebuild_states(
        te_states_base, te_devs, BW_max, C_edge_max)

    # Load agent
    agent = MultiDeviceAgent()
    if ckpt_path and os.path.exists(ckpt_path):
        agent.online.load_state_dict(
            torch.load(ckpt_path, map_location="cpu"))
        agent.online.eval()
    else:
        print(f"  ⚠ Checkpoint not found: {ckpt_path}")

    results = {}
    for mname in ["SemCIDS","Fixed-Semantic",
                  "Always-Offload"]:
        accs=[]; lats=[]; offs=[]; ks=[]; bws=[]

        for i in range(len(te_st)):
            s=te_st[i]; devs=te_devs[i]
            C_edge=get_C_edge_val(devs,C_edge_max)
            BW_t  =get_BW_val(devs,BW_max)

            if mname=="SemCIDS":
                with torch.no_grad():
                    q=agent.online(
                        torch.FloatTensor(s).unsqueeze(0))
                actions=[ACTIONS[qi.argmax().item()]
                         for qi in q]
            elif mname=="Fixed-Semantic":
                with torch.no_grad():
                    q=agent.online(
                        torch.FloatTensor(s).unsqueeze(0))
                actions=[(ACTIONS[qi.argmax().item()][0],2)
                         for qi in q]
            else:
                actions=[(1.0,0)]*N_DEVICES

            sa=[]; sl=[]; so=[]; sk=[]; bw_s=0.0
            for di,(O,K) in enumerate(actions):
                t=devs[di]
                C_l=get_C_local(t["intensity"])
                acc=detection_accuracy(t,O,K)
                lat=compute_latency_simple(
                    t,O,K,C_l,BW_t,C_edge,
                    devs,actions,T_max)
                sa.append(acc); sl.append(lat)
                so.append(O);   sk.append(K)
                bw_s+=O*COMPRESSION[K]["ratio"]

            accs.append(float(np.mean(sa)))
            lats.append(float(np.mean(sl)))
            offs.append(float(np.mean(so)))
            ks.append(float(np.mean(sk)))
            bws.append(bw_s/N_DEVICES)

        results[mname]=dict(
            accuracy     =float(np.mean(accs)),
            latency_ms   =float(np.mean(lats))*1000,
            offload_ratio=float(np.mean(offs)),
            avg_comp_k   =float(np.mean(ks)),
            bw_used_pct  =float(np.mean(bws))*100)
        print(f"    {mname:<18} "
              f"acc={results[mname]['accuracy']*100:5.1f}% "
              f"off={results[mname]['offload_ratio']:.2f} "
              f"K={results[mname]['avg_comp_k']:.2f} "
              f"lat={results[mname]['latency_ms']:.2f}ms")
    return results

# ── Experiment settings ───────────────────────────────────────
EXPERIMENTS = [
    # (exp_name, BW, C_edge, T_max, suffix)
    *[("vary_bw",    bw,   260.0, 10.0,
       f"bw{int(bw)}")
      for bw in [10,15,20,25,35,45]],
    *[("vary_cedge", 45.0, ce,    10.0,
       f"ce{int(ce)}")
      for ce in [130,200,260]],
    *[("vary_tmax",  45.0, 260.0, tm,
       f"tmax{int(tm)}")
      for tm in [5.0,10.0,15.0]],
]

# ── Run all settings ──────────────────────────────────────────
rows = []
for exp,bw,ce,tm,suffix in EXPERIMENTS:
    print(f"\n[11] {exp}: BW={bw} Ce={ce} Tm={tm} ...")

    # Train via 05_train.py
    ckpt = train_setting(bw, ce, tm,
                         suffix, n_epochs=15)

    # Evaluate
    res = evaluate_setting(ckpt, bw, ce, tm)
    for mname,r in res.items():
        rows.append({
            "experiment": exp,
            "bw_mbps":    bw,
            "c_edge":     ce,
            "t_max":      tm,
            "suffix":     suffix,
            "method":     mname,
            **r})

df = pd.DataFrame(rows)
df.to_csv("results/bandwidth_experiment.csv",
          index=False)
print(f"\n[11] Saved → results/bandwidth_experiment.csv")
print(f"     {len(df)} rows")

# ── Plot (same as 11b_plot_bandwidth.py) ─────────────────────
print("[11] Generating plots ...")
import importlib.util, pathlib
plot_script = "11b_plot_bandwidth.py"
if os.path.exists(plot_script):
    subprocess.run([sys.executable, plot_script])
else:
    print(f"  ⚠ {plot_script} not found — skipping plots")
    print("    Run 11b_plot_bandwidth.py separately")

print("[11] Complete.")
