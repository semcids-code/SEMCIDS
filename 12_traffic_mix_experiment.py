"""
12_traffic_mix_experiment.py
"""
import os, sys, pickle, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,".")
from drl_agent import (MultiDeviceAgent, ACTIONS,
                        COMPRESSION, N_DEVICES)

SEED=42; random.seed(SEED); np.random.seed(SEED)
os.makedirs("results",exist_ok=True)
os.makedirs("Images", exist_ok=True)

cfg      = pickle.load(open("results/config.pkl","rb"))
N        = cfg["N_DEVICES"]
ALPHA_C  = cfg.get("ALPHA_C",   45.0)
C_EDGE_MAX=cfg.get("C_EDGE_MAX",260.0)
ALPHA_E_ = cfg.get("ALPHA_E",   80.0)
ALPHA_B  = cfg.get("ALPHA_B",   30.0)
BW_MAX   = cfg.get("BW_MAX",    45.0)

X_test = np.load("results/X_test.npy")
y_test = np.load("results/y_test.npy")
p_test = np.load("results/p_test.npy")
te_states=np.load("results/test_states.npy")
te_devs  =pickle.load(open("results/test_devices.pkl","rb"))

def load_agent(ckpt):
    a=MultiDeviceAgent()
    if os.path.exists(ckpt):
        a.online.load_state_dict(
            torch.load(ckpt,map_location="cpu"))
        a.online.eval()
        print(f"  ✓ {ckpt}")
    else:
        print(f"  ⚠ {ckpt} not found")
    return a

agent_prop = load_agent("checkpoints/dqn_proposed.pth")
agent_nopri= load_agent("checkpoints/dqn_no_priority.pth")

def get_C_local(xi):
    return float(np.clip(75.0-ALPHA_C*xi,30.0,75.0))
def get_C_edge(devs):
    avg=float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(
        C_EDGE_MAX-ALPHA_E_*avg,C_EDGE_MAX*0.3,C_EDGE_MAX))
def detection_accuracy(t,O,K):
    af=COMPRESSION[K]["acc_f"]; xi=t["intensity"]
    pri=t["priority"]
    pb={2:1.0,1:0.95,0:0.90}.get(int(pri),0.95)
    return float(np.clip(
        (1-O)*(0.88-0.50*xi)
        +O*min(1.0,(0.76+0.23*xi)*af*pb),0,1))

# ── Build mixed dataset at given attack fraction ──────────────
def build_mixed_indices(attack_frac, n_total=1000):
    """
    Sample n_total indices from test set with
    given fraction of attack flows.
    """
    atk_idx = np.where(y_test==1)[0]
    ben_idx = np.where(y_test==0)[0]
    n_atk = int(n_total*attack_frac)
    n_ben = n_total - n_atk
    n_atk = min(n_atk, len(atk_idx))
    n_ben = min(n_ben, len(ben_idx))
    chosen_atk=np.random.choice(atk_idx,n_atk,
                                 replace=False)
    chosen_ben=np.random.choice(ben_idx,n_ben,
                                 replace=False)
    idx=np.concatenate([chosen_atk,chosen_ben])
    np.random.shuffle(idx)
    return idx

def evaluate_mix(agent, indices):
    accs=[]; offloads=[]; comp_ks=[]; bw_used=[]
    for i in indices:
        s=te_states[i]; devs=te_devs[i]
        with torch.no_grad():
            q=agent.online(
                torch.FloatTensor(s).unsqueeze(0))
        actions=[ACTIONS[qi.argmax().item()] for qi in q]
        step_acc=[]; step_o=[]; step_k=[]
        bw_step=0.0
        for di,(O,K) in enumerate(actions):
            t=devs[di]
            acc=detection_accuracy(t,O,K)
            step_acc.append(acc)
            step_o.append(O)
            step_k.append(K)
            bw_step+=O*COMPRESSION[K]["ratio"]
        accs.append(float(np.mean(step_acc)))
        offloads.append(float(np.mean(step_o)))
        comp_ks.append(float(np.mean(step_k)))
        bw_used.append(bw_step/N_DEVICES)
    return dict(
        accuracy     =float(np.mean(accs)),
        offload_ratio=float(np.mean(offloads)),
        avg_comp_k   =float(np.mean(comp_ks)),
        bw_pct       =float(np.mean(bw_used))*100,
    )

ATTACK_FRACS=[0.10,0.30,0.50,0.70,0.90]
N_TOTAL=1000

print("[12] Running traffic mix experiment ...")
rows=[]
for frac in ATTACK_FRACS:
    np.random.seed(SEED)
    idx=build_mixed_indices(frac,N_TOTAL)
    for mname,agent in [
        ("SemCIDS (Proposed)", agent_prop),
        ("No Priority",        agent_nopri),
    ]:
        res=evaluate_mix(agent,idx)
        rows.append({
            "attack_frac_pct": frac*100,
            "method":          mname,
            **res})
        print(f"  atk={frac*100:.0f}% {mname:<22} "
              f"acc={res['accuracy']*100:5.1f}% "
              f"off={res['offload_ratio']:.3f} "
              f"K={res['avg_comp_k']:.2f} "
              f"bw={res['bw_pct']:.1f}%")

df=pd.DataFrame(rows)
df.to_csv("results/traffic_mix_experiment.csv",
          index=False)
print("[12] Saved → results/traffic_mix_experiment.csv")

# ── Plot: 4 subplots ──────────────────────────────────────────
MCOLS={"SemCIDS (Proposed)":"#C03030",
        "No Priority":"#3060C0"}
MLST ={"SemCIDS (Proposed)":"-",
        "No Priority":"--"}
MMRK ={"SemCIDS (Proposed)":"o",
        "No Priority":"s"}

metrics=[
    ("accuracy",     "Detection Accuracy (%)",
     lambda x:x*100),
    ("offload_ratio","Offload Ratio",
     lambda x:x),
    ("avg_comp_k",   "Avg Compression Level K",
     lambda x:x),
    ("bw_pct",       "Bandwidth Used (%)",
     lambda x:x),
]

fig,axs=plt.subplots(2,2,figsize=(14,10))
fig.suptitle(
    "Experiment 2: Behaviour Across Traffic Mix\n"
    "SemCIDS adapts offloading and compression "
    "as attack fraction increases",
    fontsize=12,fontweight="bold")

for ax,(col,yl,fn) in zip(axs.flat,metrics):
    for m in ["SemCIDS (Proposed)","No Priority"]:
        ms=df[df.method==m].sort_values(
            "attack_frac_pct")
        ax.plot(ms["attack_frac_pct"],
                fn(ms[col]),
                color=MCOLS[m],ls=MLST[m],lw=2.5,
                marker=MMRK[m],ms=8,
                label=m)
    ax.set_xlabel("Attack Fraction (%)",fontsize=11)
    ax.set_ylabel(yl,fontsize=11)
    ax.set_xticks([10,30,50,70,90])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3,linestyle="--")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8)

plt.tight_layout()
plt.savefig("Images/IDS_traffic_mix_experiment.png",
            dpi=150,bbox_inches="tight")
plt.close()
print("[12] Saved → Images/IDS_traffic_mix_experiment.png")
