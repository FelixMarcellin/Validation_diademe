import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DIR_G = Path(r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe\DATA\gouttiere\tous fichiers gouttiere")
DIR_D = Path(r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe\DATA\diademe\tous fichiers Diademe")
OUT = Path(r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe\RESULTATS_COMPARAISON_V4.1")
OUT.mkdir(parents=True, exist_ok=True)

REF_POINTS = ['M1', 'M2', 'M3']

def get_local_basis(m1, m2, m3):
    try:
        if np.isnan(m1).any() or np.isnan(m2).any() or np.isnan(m3).any():
            return None, None
        origin = m1
        vx = m2 - m1
        norm_x = np.linalg.norm(vx)
        if norm_x < 1e-6: return None, None
        vx /= norm_x
        v_temp = m3 - m1
        vz = np.cross(vx, v_temp)
        norm_z = np.linalg.norm(vz)
        if norm_z < 1e-6: return None, None
        vz /= norm_z
        vy = np.cross(vz, vx)
        return np.stack([vx, vy, vz], axis=1), origin
    except: return None, None

def load_vicon(p):
    try:
        df = pd.read_csv(p, skiprows=4, low_memory=False).apply(pd.to_numeric, errors='coerce')
        with open(p, "r", encoding="utf-8") as f:
            ls = f.readlines()
            names = [n.split(":")[-1].strip() for n in ls[2].strip().split(",") if n.strip()]
        data = {n: df.iloc[:, i*3+2:i*3+5].values for i, n in enumerate(names) if n}
        return data
    except: return None

# --- PROCESSING ---
all_results = []
for f_g_path in DIR_G.glob("*.csv"):
    data_g = load_vicon(f_g_path)
    f_d_path = DIR_D / f_g_path.name.replace("_G_", "_D_")
    if data_g and f_d_path.exists():
        data_d = load_vicon(f_d_path)
        mvt = "M" + f_g_path.name.split("_M")[1][0] if "_M" in f_g_path.name else "M?"
        facials = [m for m in data_g.keys() if m not in REF_POINTS and m in data_d]
        for m in facials:
            n_frames = min(len(data_g[m]), len(data_d[m]))
            p_ds_loc, p_hb_loc = [], []
            for f in range(n_frames):
                R_g, org_g = get_local_basis(data_g['M1'][f], data_g['M2'][f], data_g['M3'][f])
                R_d, org_d = get_local_basis(data_d['M1'][f], data_d['M2'][f], data_d['M3'][f])
                if R_g is not None and R_d is not None:
                    p_ds_loc.append(R_g.T @ (data_g[m][f] - org_g))
                    p_hb_loc.append(R_d.T @ (data_d[m][f] - org_d))
            if len(p_ds_loc) > 10:
                p_ds_loc, p_hb_loc = np.array(p_ds_loc), np.array(p_hb_loc)
                p_ds_rel, p_hb_rel = p_ds_loc - p_ds_loc[0], p_hb_loc - p_hb_loc[0]
                err_3d = np.linalg.norm(p_ds_rel - p_hb_rel, axis=1)
                d_ds, d_hb = np.linalg.norm(p_ds_rel, axis=1), np.linalg.norm(p_hb_rel, axis=1)
                r_corr = pearsonr(d_ds, d_hb)[0] if np.std(d_ds)>1e-6 and np.std(d_hb)>1e-6 else 0
                all_results.append({
                    "File": f_g_path.name, "Movement": mvt, "Marker": m,
                    "Pearson_R": abs(r_corr), "MAE_mm": np.nanmean(err_3d),
                    "ROM_Ds": np.ptp(d_ds), "ROM_Hb": np.ptp(d_hb),
                    "Dist_Ds": d_ds, "Dist_Hb": d_hb 
                })

df = pd.DataFrame(all_results).dropna(subset=['MAE_mm'])

# --- FIGURES ---
plt.rcParams.update({'font.size': 11})

# FIG 1 : Bland-Altman avec valeurs sur les lignes
plt.figure(figsize=(9, 6))
means, diffs = (df["ROM_Ds"] + df["ROM_Hb"]) / 2, df["ROM_Ds"] - df["ROM_Hb"]
bias, sd = np.mean(diffs), np.std(diffs)
plt.scatter(means, diffs, alpha=0.4, color='navy', s=20)
plt.axhline(bias, color='red', lw=2)
plt.axhline(bias + 1.96*sd, color='gray', ls='--')
plt.axhline(bias - 1.96*sd, color='gray', ls='--')

# Affichage des valeurs à l'extrémité droite du graphique
x_pos = plt.xlim()[1]
plt.text(x_pos, bias, f' Bias: {bias:.3f}', color='red', va='center', fontweight='bold')
plt.text(x_pos, bias + 1.96*sd, f' +1.96SD: {bias + 1.96*sd:.3f}', color='gray', va='center')
plt.text(x_pos, bias - 1.96*sd, f' -1.96SD: {bias - 1.96*sd:.3f}', color='gray', va='center')

plt.title("Figure 1: Bland-Altman Agreement (ROM)"); plt.xlabel("Mean ROM [mm]"); plt.ylabel("Diff (Ds-Hb) [mm]")
plt.tight_layout(); plt.savefig(OUT / "Fig1_BlandAltman.png", dpi=300)

# FIG 2 : Boxplot (Réintégré)
plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x="Movement", y="MAE_mm", palette="Blues", width=0.6)
plt.title("Figure 2: Distribution of MAE per Movement"); plt.ylabel("Mean Absolute Error [mm]")
plt.tight_layout(); plt.savefig(OUT / "Fig2_Boxplot.png", dpi=300)

# FIG 3 : Trajectory
sel = df[(df["Movement"] == "M5") & (df["Marker"] == "C02")]
if not sel.empty:
    plt.figure(figsize=(10, 5)); s = sel.iloc[0]
    plt.plot(s["Dist_Ds"], 'k-', label="Dental Splint (Ds)", lw=2)
    plt.plot(s["Dist_Hb"], 'r--', label="Headband (Hb)", lw=2)
    plt.xlabel("Time [Frames]"); plt.ylabel("Displacement [mm]")
    plt.title("Figure 3: Tracking Comparison of the lest labial commissure"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "Fig3_Trajectory.png", dpi=300)

# FIG 4 : Frequency avec Médiane
plt.figure(figsize=(8, 6))
med = df["MAE_mm"].median()
sns.histplot(df["MAE_mm"], bins=50, kde=True, color='skyblue')
plt.axvline(med, color='red', ls='--', lw=2)
plt.text(med, plt.ylim()[1]*0.8, f' Median: {med:.3f} mm', color='orange', fontweight='bold')
plt.yscale('log'); plt.xlabel("MAE [mm]"); plt.title("Figure 4: Error Distribution")
plt.tight_layout(); plt.savefig(OUT / "Fig4_Frequency.png", dpi=300)

# Tableaux
df.groupby("Movement").agg({"Pearson_R": ["mean", "std"], "MAE_mm": ["mean", "std"]}).to_csv(OUT / "Table_1_Movements.csv")
df.groupby("Marker")["MAE_mm"].agg(["mean", "std"]).to_csv(OUT / "Table_2_Markers.csv")
df.groupby("File").agg({"Pearson_R": "mean", "MAE_mm": "mean"}).to_csv(OUT / "Table_3_Files_Recap.csv")

print(f"✅ Pack complet généré (4 Figures + 3 Tableaux) dans : {OUT}")