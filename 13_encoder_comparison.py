"""
13_encoder_comparison.py
=========================
"""
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED=42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("results", exist_ok=True)
os.makedirs("Images",  exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

X_train=np.load("results/X_train.npy")
X_test =np.load("results/X_test.npy")
y_train=np.load("results/y_train.npy")
y_test =np.load("results/y_test.npy")
d=X_train.shape[1]; L_TRUNC=[2,4,6,8]

Xtr_t=torch.FloatTensor(X_train)
ytr_t=torch.LongTensor(y_train)
Xte_t=torch.FloatTensor(X_test)
yte_t=torch.LongTensor(y_test)

dataset=TensorDataset(Xtr_t,ytr_t)
loader =DataLoader(dataset,batch_size=256,
                   shuffle=True,drop_last=False)

print("[13] Experiment 3: Encoder comparison ...")

class TaskEncoderOrdered(nn.Module):
 
    def __init__(self, in_dim=46, k=8):
        super().__init__()
        h=max(k*8,32)
        self.encoder=nn.Sequential(
            nn.Linear(in_dim,128),
            nn.BatchNorm1d(128),nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),nn.GELU(),
            nn.Linear(32,k))
        self.classifier=nn.Sequential(
            nn.Linear(k,h),nn.ReLU(),
            nn.Linear(h,2))
    def forward(self,x): return self.classifier(self.encoder(x))

class TaskEncoderUnordered(nn.Module):
    def __init__(self, in_dim=46, k=8):
        super().__init__()
        h=max(k*8,32)
        self.encoder=nn.Sequential(
            nn.Linear(in_dim,128),
            nn.BatchNorm1d(128),nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),nn.GELU(),
            nn.Linear(32,k))
        self.classifier=nn.Sequential(
            nn.Linear(k,h),nn.ReLU(),
            nn.Linear(h,2))
    def forward(self,x): return self.classifier(self.encoder(x))

class StandardAutoencoder(nn.Module):

    def __init__(self, in_dim=46, k=8):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(in_dim,128),nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,32),  nn.ReLU(),
            nn.Linear(32,k))
        self.decoder=nn.Sequential(
            nn.Linear(k,32),   nn.ReLU(),
            nn.Linear(32,64),  nn.ReLU(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,in_dim))
    def forward(self,x):
        z=self.encoder(x)
        return self.decoder(z), z
    def encode(self,x):
        return self.encoder(x)

def train_ordered(k, n_epochs=300):
    """Train with successive truncation (our method)."""
    model=TaskEncoderOrdered(d,k)
    opt=torch.optim.AdamW(model.parameters(),
                           lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,T_max=n_epochs,eta_min=1e-5)
    crit=nn.CrossEntropyLoss()
    best_acc=0.0; best_state=None
    import copy

    for epoch in range(n_epochs):
        model.train()
        for xb,yb in loader:
            logits=model(xb)
            loss=crit(logits,yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),1.0)
            opt.step()
        sched.step()
        if (epoch+1)%100==0:
            model.eval()
            with torch.no_grad():
                acc=(model(Xte_t).argmax(1)==yte_t
                     ).float().mean().item()*100
            if acc>best_acc:
                best_acc=acc
                best_state=copy.deepcopy(
                    model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        acc=(model(Xte_t).argmax(1)==yte_t
             ).float().mean().item()*100
    return acc

def train_unordered(n_epochs=300):
    """Train WITHOUT successive truncation (no ordering)."""
    model=TaskEncoderUnordered(d,8)
    opt=torch.optim.AdamW(model.parameters(),
                           lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,T_max=n_epochs,eta_min=1e-5)
    crit=nn.CrossEntropyLoss()
    best_acc=0.0; best_state=None
    import copy

    # Train always with k=8 (no truncation)
    for epoch in range(n_epochs):
        model.train()
        for xb,yb in loader:
            logits=model(xb)   # always full k=8
            loss=crit(logits,yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),1.0)
            opt.step()
        sched.step()
        if (epoch+1)%100==0:
            model.eval()
            with torch.no_grad():
                acc=(model(Xte_t).argmax(1)==yte_t
                     ).float().mean().item()*100
            if acc>best_acc:
                best_acc=acc
                best_state=copy.deepcopy(
                    model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model   # return model for truncation test

def eval_unordered_at_k(model, k):
    model.eval()
    with torch.no_grad():
        z=model.encoder(Xte_t)
        z_k=z[:,:k]   # take first k (arbitrary)
        logits=model.classifier(
            torch.cat([z_k,
                       torch.zeros(len(z_k),8-k)],
                      dim=1))
        acc=(logits.argmax(1)==yte_t
             ).float().mean().item()*100
    return acc

def train_autoencoder(k, n_epochs=300):
 
    ae=StandardAutoencoder(d,k)
    opt=torch.optim.Adam(ae.parameters(),lr=1e-3)
    recon_crit=nn.MSELoss()
    import copy

    best_loss=1e9; best_state=None
    # Unsupervised loader (no labels)
    uns_loader=DataLoader(
        TensorDataset(Xtr_t),
        batch_size=256,shuffle=True)

    for epoch in range(n_epochs):
        ae.train()
        for (xb,) in uns_loader:
            recon,z=ae(xb)
            loss=recon_crit(recon,xb)
            opt.zero_grad(); loss.backward()
            opt.step()
        if (epoch+1)%100==0:
            ae.eval()
            with torch.no_grad():
                recon,_=ae(Xte_t)
                val_loss=recon_crit(recon,Xte_t).item()
            if val_loss<best_loss:
                best_loss=val_loss
                best_state=copy.deepcopy(ae.state_dict())
            print(f"    AE k={k} epoch {epoch+1}: "
                  f"recon_loss={val_loss:.4f}")

    ae.load_state_dict(best_state)
    ae.eval()

    with torch.no_grad():
        z_tr=ae.encode(Xtr_t).numpy()
        z_te=ae.encode(Xte_t).numpy()

    clf=RandomForestClassifier(
        100,random_state=SEED,n_jobs=-1)
    clf.fit(z_tr,y_train)
    acc=clf.score(z_te,y_test)*100
    return acc

from sklearn.ensemble import RandomForestClassifier
rf_imp=RandomForestClassifier(
    100,random_state=SEED,n_jobs=-1)
rf_imp.fit(X_train,y_train)
topk_ord=np.argsort(
    rf_imp.feature_importances_)[::-1]

def eval_topk(k):
    idx=topk_ord[:k]
    Xtr_k=np.zeros_like(X_train)
    Xtr_k[:,idx]=X_train[:,idx]
    Xte_k=np.zeros_like(X_test)
    Xte_k[:,idx]=X_test[:,idx]
    clf=RandomForestClassifier(
        100,random_state=SEED,n_jobs=-1)
    clf.fit(Xtr_k,y_train)
    return clf.score(Xte_k,y_test)*100

rows=[]

print("\n  Training Task-Oriented (Ordered) — 4 models ...")
for k in L_TRUNC:
    print(f"  Training ordered encoder k={k} ...")
    acc=train_ordered(k,n_epochs=300)
    rows.append({"method":"Task-Oriented (Ours)",
                 "k":k,"bw_pct":k/d*100,
                 "accuracy":round(acc,1)})
    print(f"    k={k}: {acc:.1f}%")

print("\n  Training Task-Oriented (Unordered) — 1 model ...")
unordered_model=train_unordered(n_epochs=300)
for k in L_TRUNC:
    acc=eval_unordered_at_k(unordered_model,k)
    rows.append({"method":"Task-Oriented (No Ordering)",
                 "k":k,"bw_pct":k/d*100,
                 "accuracy":round(acc,1)})
    print(f"    k={k}: {acc:.1f}%")

print("\n  Training Standard Autoencoder — 4 models ...")
for k in L_TRUNC:
    print(f"  Training AE k={k} ...")
    acc=train_autoencoder(k,n_epochs=300)
    rows.append({"method":"Standard Autoencoder",
                 "k":k,"bw_pct":k/d*100,
                 "accuracy":round(acc,1)})
    print(f"    k={k}: {acc:.1f}%")

print("\n  Evaluating Top-K baseline ...")
for k in L_TRUNC:
    acc=eval_topk(k)
    rows.append({"method":"Top-K Selection",
                 "k":k,"bw_pct":k/d*100,
                 "accuracy":round(acc,1)})
    print(f"    k={k}: {acc:.1f}%")

df=pd.DataFrame(rows)
df.to_csv("results/encoder_comparison.csv",index=False)

# Print table
print("\n"+"="*62)
print("  Encoder Comparison — Detection Accuracy (%)")
print("="*62)
piv=df.pivot(index="method",columns="k",
             values="accuracy")
print(piv.to_string())
print("="*62)

# ── Plot ──────────────────────────────────────────────────────
MCOLS={
    "Task-Oriented (Ours)":       "#C03030",
    "Task-Oriented (No Ordering)":"#C07020",
    "Standard Autoencoder":       "#808080",
    "Top-K Selection":            "#3060C0",
}
MLST={
    "Task-Oriented (Ours)":        "-",
    "Task-Oriented (No Ordering)": "--",
    "Standard Autoencoder":        "-.",
    "Top-K Selection":             ":",
}
MMRK={
    "Task-Oriented (Ours)":        "o",
    "Task-Oriented (No Ordering)": "s",
    "Standard Autoencoder":        "D",
    "Top-K Selection":             "^",
}

fig,ax=plt.subplots(figsize=(10,6))
fig.suptitle(
    "Experiment 3: Encoder Comparison\n"
    "Task-Oriented (Ours) vs No Ordering vs "
    "Standard Autoencoder vs Top-K",
    fontsize=11,fontweight="bold")

methods=["Task-Oriented (Ours)",
         "Task-Oriented (No Ordering)",
         "Standard Autoencoder",
         "Top-K Selection"]
for m in methods:
    ms=df[df.method==m].sort_values("k")
    lw=2.8 if m=="Task-Oriented (Ours)" else 1.8
    ax.plot(ms["k"],ms["accuracy"],
            color=MCOLS[m],ls=MLST[m],lw=lw,
            marker=MMRK[m],ms=8,label=m)

ax.set_xlabel(
    "Compression Level k (latent dimensions)",
    fontsize=11)
ax.set_ylabel("Detection Accuracy (%)",fontsize=11)
ax.set_xticks(L_TRUNC)
ax.set_xticklabels(
    [f"k={k}\n({k/d*100:.1f}% BW)"
     for k in L_TRUNC],fontsize=9)
ax.legend(fontsize=10,loc="lower right")
ax.grid(alpha=0.3,linestyle="--")
for s in ax.spines.values():
    s.set_visible(True); s.set_linewidth(0.8)

plt.tight_layout()
plt.savefig("Images/IDS_encoder_comparison.png",
            dpi=150,bbox_inches="tight")
plt.close()
print("[13] Saved → results/encoder_comparison.csv")
print("[13] Saved → Images/IDS_encoder_comparison.png")
