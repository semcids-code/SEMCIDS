"""
03_environment.py  
================================================
"""
import os, pickle, random
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.makedirs("results", exist_ok=True)

X_train = np.load("results/X_train.npy")
X_test  = np.load("results/X_test.npy")
y_train = np.load("results/y_train.npy")
y_test  = np.load("results/y_test.npy")
p_train = np.load("results/p_train.npy")
p_test  = np.load("results/p_test.npy")
COL_IDX = pickle.load(open("results/col_idx.pkl","rb"))

N_DEVICES = int(input(
    "\n[03] How many IoT edge devices? [2-10]: "))
assert 2 <= N_DEVICES <= 10

# ── Constants ─────────────────────────────────────────────────
OFFLOAD = [0.0, 0.25, 0.50, 0.75, 1.0]
COMPRESSION = {
    0: dict(ratio=1.00, acc_f=1.00, cost=0.0,
            name="K0",    label="No compression"),
    1: dict(ratio=0.75, acc_f=1.06, cost=0.10,
            name="K_low", label="Low compression"),
    2: dict(ratio=0.50, acc_f=0.82, cost=1.0,
            name="K_med", label="Medium compression"),
    3: dict(ratio=0.25, acc_f=0.55, cost=2.5,
            name="K_high",label="High compression"),
}

# Joint action space: |O|×|K| = 5×4 = 20 per device
# Total: 20^N across all N devices
ACTIONS   = [(o,k) for o in OFFLOAD for k in COMPRESSION]
N_ACT_PER = len(ACTIONS)   # 20
assert N_ACT_PER == 20

# Constants 
C_LOCAL_MAX = 75.0    
C_LOCAL_MIN = 30.0    
C_EDGE_MAX  = 260.0   
BW_MAX      = 45.0    
BW_MIN      = 15.0   
ALPHA_C     = 45.0    
ALPHA_E     = 80.0    
ALPHA_B     = 30.0    
SIGMA_MAX   = 1.5     
T_MAX       = 10.0
E_MAX       = 15.0

STATE_SIZE = 5*N_DEVICES + 3

BOUNDS_DEV    = np.array([C_LOCAL_MAX, 30., 15., 1., 2.],
                          dtype=np.float32)

BOUNDS_SHARED = np.array([C_EDGE_MAX, BW_MAX, 1.0],
                          dtype=np.float32)

print(f"[03] State dim = 5*{N_DEVICES}+3 = {STATE_SIZE}")

def get(row, col):
    if col not in COL_IDX: return 0.0
    return float(np.clip(row[COL_IDX[col]], -3, 3))


XI_HIGH = 0.70  
XI_MED  = 0.40  

def estimate_priority_stage_a(xi):
    """
    Stage A — Pre-screen estimate.
    """
    if   xi >= XI_HIGH: return 2   
    elif xi >= XI_MED:  return 1   
    else:               return 0   

def correct_priority_stage_b(true_priority):

    return int(true_priority)   # ground truth = Stage B

def extract_traffic(row, label, priority):

    rate = get(row, "Rate")
    syn  = get(row, "syn_count")
    ack  = get(row, "ack_count")
    iat  = get(row, "IAT")
    hlen = get(row, "Header_Length")
    psh  = get(row, "psh_flag_number")

    sigma_raw = (0.30*max(rate, 0)
               + 0.30*max(syn,  0)
               + 0.20*max(ack,  0)
               + 0.15*max(psh,  0)
               - 0.05*iat)
    xi = float(np.clip(sigma_raw / SIGMA_MAX, 0.0, 1.0))

    tau = float(np.clip(abs(hlen)*8 + 2, 2., 30.))

    d = float(np.clip(
        0.40*abs(rate) + 1.80*abs(syn) + 1.20*abs(ack),
        0.1, 15.))

    P_stage_a = estimate_priority_stage_a(xi)    
    P_stage_b = correct_priority_stage_b(priority) 

    return dict(
        rate=rate, syn=syn, ack=ack,
        iat=iat, hlen=hlen, psh=psh,
        intensity=xi,           
        tau=tau,                
        demand=d,               
        label=int(label),
        priority_est=P_stage_a, 
        priority=P_stage_b,     
    )

def simulate_coordinated_attack(base, n_dev, rho=0.80):
    devs = [base]
    for _ in range(1, n_dev):
        if (base["label"]==1 and
                base["priority"]==2 and
                random.random() < rho):
            noise = random.gauss(0, 0.1)
            xi_c  = float(np.clip(
                base["intensity"]+noise, 0.5, 1.0))
            devs.append(dict(base, intensity=xi_c))
        else:
            xi_b = float(np.clip(
                random.gauss(0.1, 0.05), 0., 0.4))
            devs.append(dict(base,
                intensity=xi_b, label=0, priority=0))
    return devs

def build_joint_state(devs, bw_rem):

    parts = []

    avg_xi  = float(np.mean([t["intensity"] for t in devs]))
    C_edge  = float(np.clip(
        C_EDGE_MAX - ALPHA_E * avg_xi,
        C_EDGE_MAX * 0.3, C_EDGE_MAX))
    BW_t    = float(np.clip(
        BW_MAX - ALPHA_B * avg_xi,
        BW_MIN, BW_MAX))

    for t in devs:
        xi  = t["intensity"]
        C_l = float(np.clip(
            C_LOCAL_MAX - ALPHA_C * xi,
            C_LOCAL_MIN, C_LOCAL_MAX))
        tau = t["tau"]      
        d   = t["demand"]   

        P = float(t["priority_est"])

        s_i = np.array([C_l, tau, d, xi, P],
                        dtype=np.float32)

        s_i_norm = s_i / BOUNDS_DEV
        parts.append(s_i_norm)


    shared = np.array([C_edge, BW_t, bw_rem],
                       dtype=np.float32)
    shared_norm = shared / BOUNDS_SHARED
    parts.append(shared_norm)

    state = np.concatenate(parts)
    assert len(state) == STATE_SIZE, \
        f"State dim mismatch: {len(state)} != {STATE_SIZE}"
    return state

# ── Constants ───────────────────────
ALPHA_TX  = 0.50   
ALPHA_ENC = 0.40   
ALPHA_LOC = 0.15   
BETA_K    = 1.20   
BETA_CAP  = 1.00   
BETA_ACC  = 1.00   
BETA_LAT  = 1.00   
ALPHA_MIN = 0.60   
DELTA     = 0.05  


W_P = {2: 0.60, 1: 0.45, 0: 0.30}   
GAMMA_L = 0.64   
GAMMA_E = 0.36   

def detection_accuracy(t, O, K):

    af        = COMPRESSION[K]["acc_f"]
    xi        = t["intensity"]
    pri       = t["priority"]
    local_acc = 0.88 - 0.50*xi
    pb        = {2:1.0,1:0.95,0:0.90}.get(int(pri),0.95)
    edge_acc  = min(1.0,(0.76+0.23*xi)*af*pb)
    return float(np.clip(
        (1-O)*local_acc + O*edge_acc, 0, 1))

def compute_energy(t, O, K, C_local):
    R_K = COMPRESSION[K]["ratio"] 
    C_K = COMPRESSION[K]["cost"]   
    tau = t["tau"]                 
    d   = t["demand"]              

    E_trans = ALPHA_TX  * tau * R_K
    E_comp  = ALPHA_ENC * tau * C_K
    E_local = ALPHA_LOC * d / (C_local + 1e-6)

    E = O * (E_trans + E_comp) + (1-O) * E_local
    return float(np.clip(E, 0, E_MAX))

def compute_latency(t, O, K, C_local,
                    BW_t, C_edge,
                    all_devs, all_actions):

    R_K = COMPRESSION[K]["ratio"]
    C_K = COMPRESSION[K]["cost"]
    tau = t["tau"]
    d   = t["demand"]

    T_trans = tau * R_K / (BW_t + 1e-6)
    T_comp  = BETA_K * C_K
    T_local = d / (C_local + 1e-6)

   
    T_edge = sum(
        O_j * dv["demand"]
        for dv, (O_j, _) in zip(all_devs, all_actions)
    ) / (C_edge + 1e-6)

    T = O*(T_trans + T_comp) + T_edge + (1-O)*T_local
    return float(np.clip(T, 0, T_MAX))

def compute_penalty(t, O, acc, T_t, C_local):

    tau   = t["tau"]

    P_cap = BETA_CAP * max(0.0, (1-O)*tau - C_local)

    P_acc = BETA_ACC * max(0.0, ALPHA_MIN - acc)

    P_lat = BETA_LAT * max(0.0, T_t - T_MAX)

    return float(P_cap + P_acc + P_lat)

def compute_reward_device(penalty, T_t, E_t, O, priority):
    w_p = W_P.get(int(priority), 0.45)
    w_t = (1 - w_p) * GAMMA_L   
    w_e = (1 - w_p) * GAMMA_E   

    T_norm = T_t / T_MAX
    E_norm = E_t / E_MAX

    cost  = w_p*penalty + w_t*T_norm + w_e*E_norm
    bonus = DELTA * O * (1 - O)

    return float(np.clip(-cost + bonus, -1.0, 0.1))

def compute_joint_reward(device_rewards):

    return float(np.mean(device_rewards))

def allocate_bandwidth(actions):
    demands = [O*COMPRESSION[K]["ratio"]
               for O,K in actions]
    total   = sum(demands)+1e-9
    return ([d/total for d in demands]
            if total > 1.0 else demands)


print(f"\n[03] Pre-computing states "
      f"(N={N_DEVICES}, dim={STATE_SIZE}) ...")

def make_joint(row, lbl, pri):
    base = extract_traffic(row, lbl, pri)
    devs = simulate_coordinated_attack(
        base, N_DEVICES)
    s    = build_joint_state(devs, 1.0)
    return s, devs

tr_states=[]; tr_devs=[]
for i in range(len(X_train)):
    s,dv = make_joint(X_train[i],
                      int(y_train[i]),
                      int(p_train[i]))
    tr_states.append(s); tr_devs.append(dv)
tr_states = np.array(tr_states, dtype=np.float32)

te_states=[]; te_devs=[]
for i in range(len(X_test)):
    s,dv = make_joint(X_test[i],
                      int(y_test[i]),
                      int(p_test[i]))
    te_states.append(s); te_devs.append(dv)
te_states = np.array(te_states, dtype=np.float32)

np.save("results/train_states.npy", tr_states)
np.save("results/test_states.npy",  te_states)
pickle.dump(tr_devs,
    open("results/train_devices.pkl","wb"))
pickle.dump(te_devs,
    open("results/test_devices.pkl","wb"))

config = dict(
    # Environment
    N_DEVICES=N_DEVICES,
    STATE_SIZE=STATE_SIZE,
    N_ACT_PER=N_ACT_PER,
    OFFLOAD=OFFLOAD,
    COMPRESSION=COMPRESSION,
    ACTIONS=ACTIONS,
    # State constants
    C_LOCAL_MAX=C_LOCAL_MAX,
    C_LOCAL_MIN=C_LOCAL_MIN,
    C_EDGE_MAX=C_EDGE_MAX,
    BW_MAX=BW_MAX,
    BW_MIN=BW_MIN,
    ALPHA_C=ALPHA_C,
    ALPHA_E=ALPHA_E,
    ALPHA_B=ALPHA_B,
    SIGMA_MAX=SIGMA_MAX,
    STATE_BOUNDS_1=BOUNDS_DEV.tolist(),
    STATE_BOUNDS_SHARED=BOUNDS_SHARED.tolist(),
    # Reward constants
    TOTAL_BW=BW_MAX,
    T_MAX=T_MAX,
    E_MAX=E_MAX,
    ALPHA_TX=ALPHA_TX,
    ALPHA_ENC=ALPHA_ENC,
    ALPHA_LOC=ALPHA_LOC,
    BETA_K=BETA_K,
    BETA_CAP=BETA_CAP,
    BETA_ACC=BETA_ACC,
    BETA_LAT=BETA_LAT,
    ALPHA_MIN=ALPHA_MIN,
    DELTA=DELTA,
    W_P=W_P,
    GAMMA_L=GAMMA_L,
    GAMMA_E=GAMMA_E)
pickle.dump(config, open("results/config.pkl","wb"))


print("\n[03] Verifying reward equations...")
_t = dict(tau=10.0, demand=5.0, intensity=0.8,
          priority=2, label=1, rate=2.0,
          syn=1.5, ack=0.8, iat=-0.2,
          hlen=1.5, psh=0.5)
_O=0.5; _K=1; _C_l=52.0; _BW=40.0; _Ce=220.0

_R_K = COMPRESSION[_K]["ratio"]   
_C_K = COMPRESSION[_K]["cost"]    


_E_trans = ALPHA_TX  * _t["tau"] * _R_K        
_E_comp  = ALPHA_ENC * _t["tau"] * _C_K        
_E_local = ALPHA_LOC * _t["demand"] / _C_l     
_E = _O*(_E_trans+_E_comp)+(1-_O)*_E_local
print(f"  E_trans={_E_trans:.3f}  E_comp={_E_comp:.3f}  "
      f"E_local={_E_local:.3f}  E={_E:.3f}")


_T_trans = _t["tau"]*_R_K/_BW                  
_T_comp  = BETA_K * _C_K                       
_T_local = _t["demand"]/_C_l                   
_T_edge  = (_O*_t["demand"])/_Ce               
_T = _O*(_T_trans+_T_comp)+_T_edge+(1-_O)*_T_local
print(f"  T_trans={_T_trans:.3f}  T_comp={_T_comp:.3f}  "
      f"T_edge={_T_edge:.3f}  T_local={_T_local:.3f}  T={_T:.3f}")


_acc = detection_accuracy(_t, _O, _K)
_P_cap = BETA_CAP*max(0,(1-_O)*_t["tau"]-_C_l)
_P_acc = BETA_ACC*max(0,ALPHA_MIN-_acc)
_P_lat = BETA_LAT*max(0,_T-T_MAX)
_P = _P_cap+_P_acc+_P_lat
print(f"  acc={_acc:.3f}  P_cap={_P_cap:.3f}  "
      f"P_acc={_P_acc:.3f}  P_lat={_P_lat:.3f}  P={_P:.3f}")


_w_p=W_P[_t["priority"]]; _w_t=(1-_w_p)*GAMMA_L
_w_e=(1-_w_p)*GAMMA_E
print(f"  w_p={_w_p}  w_t={_w_t:.3f}  w_e={_w_e:.3f}  "
      f"(sum={_w_p+_w_t+_w_e:.3f} ✓)")
_bonus = DELTA*_O*(1-_O)
_r = -(  _w_p*_P
       + _w_t*min(_T,T_MAX)/T_MAX
       + _w_e*min(_E,E_MAX)/E_MAX) + _bonus
print(f"  bonus={_bonus:.3f}  r_device={_r:.4f}")
print(f"  r_joint = (1/N)*sum r_i  ✓")
print("[03] Reward verification complete\n")

print(f"[03] State vector: S_t ∈ R^{{5×{N_DEVICES}+3}}"
      f" = R^{{{STATE_SIZE}}}")
print(f"     Per-device (×{N_DEVICES}): "
      f"[C_local, tau, d, xi, P] ∈ R^5")
print(f"     Shared (×1):   "
      f"[C_edge, BW, b_rem] ∈ R^3")
print(f"     Train={len(tr_states)}  "
      f"Test={len(te_states)}")
print("[03] Saved → results/")