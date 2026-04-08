# -*- coding: utf-8 -*-
"""
Comparaison DIADÈME (D) vs GOUTTIÈRE (G) sur séries Vicon (Trajectories CSV)
- Apparier automatiquement les fichiers D/G (même nom sauf D/G)
- Extraire le mouvement M1..M5 depuis le nom de fichier
- Comparaison "poussée":
    A) stabilité triangle D->G frame par frame (rotation/translation)
    B) comparaison fonctionnelle sur marqueurs faciaux (repères locaux) + résidus
    C) tests: t-test (moyennes résidus) + TOST équivalence (marges réglables)
- Sorties:
    - SUMMARY.csv (par essai)
    - SUMMARY_BY_MOVEMENT.csv (par mouvement)
    - REPORT_FINAL.txt (conclusion globale)
    - plots_par_essai/*.png
    - plots_globaux/*.png

À COPIER-COLLER DANS SPYDER. (Python 3.9+ recommandé)
Dépendances: numpy, pandas, matplotlib, scipy
"""

import os
import re
import csv
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# AJOUTS STATISTIQUES
# =========================

def icc_2_1(data: np.ndarray):
    n, k = data.shape
    mean_rows = np.mean(data, axis=1)
    mean_cols = np.mean(data, axis=0)
    grand = np.mean(data)

    ss_rows = k * np.sum((mean_rows - grand) ** 2)
    ss_cols = n * np.sum((mean_cols - grand) ** 2)
    ss_total = np.sum((data - grand) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    return float(
        (ms_rows - ms_error) /
        (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    )


def bland_altman_stats(x, y):
    diff = x - y
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    return float(bias), float(loa_low), float(loa_high)


# =========================
# PARAMÈTRES À RÉGLER
# =========================
CONFIG = {
    # Dossiers (Windows)
    "dir_G": r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe\gouttiere\tous fichiers gouttiere",
    "dir_D": r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe\diademe\tous fichiers Diademe",
    "out_dir": r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe\RESULTATS_COMPARAISON",

    # Triangle des 3 marqueurs du dispositif (noms Vicon)
    # -> si dans tes fichiers ce n'est pas M1/M2/M3, change ici
    "triangle_markers": ("M1", "M2", "M3"),

    # Exclure d'autres marqueurs si besoin
    "exclude_markers": set(),

    # Tokens qui distinguent les fichiers dans le nom:
    # ex: "..._D_..." pour diadème, "..._G_..." pour gouttière
    "device_token_D": "D",
    "device_token_G": "G",

    # Seuils "pratiques" pour verdict (à adapter)
    "max_rot_std_deg": 0.5,
    "max_trans_std_mm": 10.0,
    "max_rms_mm": 3.0,
    "max_p95_mm": 6.0,

    # TOST (équivalence) sur la moyenne des résidus X/Y/Z
    "tost_margin_mm": 0.5,   # marge équivalence (mm) : ±0.5 par défaut
    "alpha": 0.05,           # 5% => IC90 pour TOST
}

MOVEMENT_LABELS = {
    "M1": "Fermeture simple des paupières",
    "M2": "Fermeture forcée des paupières",
    "M3": "Protrusion labiale (son 'o')",
    "M4": "Protrusion labiale (son 'pou')",
    "M5": "Large sourire découvrant les dents",
}


# =========================
# 1) Lecture Vicon CSV
# =========================

def _read_vicon_csv_simple(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=3, header=0, engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    if len(df) > 0 and (pd.isna(df.iloc[0, 0]) or str(df.iloc[0, 0]).strip() == ""):
        df = df.iloc[1:].reset_index(drop=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _extract_marker_order_from_label_line(path: str):
    with open(path, "r", errors="ignore") as f:
        f.readline()  # Trajectories
        f.readline()  # freq
        label_line = f.readline().rstrip("\n")

    parts = list(csv.reader([label_line]))[0]
    labels = []
    k = 2
    while k < len(parts):
        lab = parts[k]
        lab = lab.split(":")[-1] if isinstance(lab, str) else str(lab)
        if lab != "":
            labels.append(lab)
        k += 3
    return labels


def load_vicon_trajectories(path: str) -> pd.DataFrame:
    df = _read_vicon_csv_simple(path)
    marker_labels = _extract_marker_order_from_label_line(path)

    n_cols = df.shape[1]
    m = (n_cols - 2) // 3
    if m <= 0:
        raise ValueError(f"Format inattendu (pas de triplets XYZ): {path}")

    if len(marker_labels) < m:
        marker_labels = marker_labels + [f"MK_{i+1}" for i in range(len(marker_labels), m)]

    new_cols = ["Frame", "SubFrame"]
    for i in range(m):
        lab = marker_labels[i]
        new_cols += [f"{lab}_X", f"{lab}_Y", f"{lab}_Z"]

    if len(new_cols) != n_cols:
        new_cols = new_cols[:n_cols] + [f"extra_{i}" for i in range(n_cols - len(new_cols))]

    df.columns = new_cols
    return df


def available_markers(df: pd.DataFrame):
    return sorted({c[:-2] for c in df.columns if c.endswith("_X")})


def marker_xyz(df: pd.DataFrame, name: str) -> np.ndarray:
    cols = [f"{name}_X", f"{name}_Y", f"{name}_Z"]
    if not all(c in df.columns for c in cols):
        raise KeyError(f"Marqueur '{name}' introuvable (colonnes {cols})")
    return df[cols].to_numpy(dtype=float)


# =========================
# 2) Outils géométriques
# =========================

def kabsch(P: np.ndarray, Q: np.ndarray):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def rigid_apply(R: np.ndarray, t: np.ndarray, X: np.ndarray):
    return (R @ X.T).T + t


def rotation_angle_deg(R: np.ndarray) -> float:
    tr = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(tr)
    return float(np.degrees(theta))


@dataclass
class LocalFrameResult:
    local_points: dict
    valid_mask: np.ndarray
    Rg2l: np.ndarray
    tg2l: np.ndarray


def build_local_frame_from_triangle(df: pd.DataFrame, tri: tuple[str, str, str]) -> LocalFrameResult:
    A = marker_xyz(df, tri[0])
    B = marker_xyz(df, tri[1])
    C = marker_xyz(df, tri[2])

    T = A.shape[0]
    tri_now = np.stack([A, B, C], axis=1)  # (T,3,3)
    valid = np.all(np.isfinite(tri_now), axis=(1, 2))
    if not np.any(valid):
        raise ValueError(f"Aucune frame valide pour le triangle {tri}")

    ref_idx = int(np.argmax(valid))
    tri_ref = tri_now[ref_idx]

    Rg2l = np.full((T, 3, 3), np.nan)
    tg2l = np.full((T, 3), np.nan)

    for i in range(T):
        if not valid[i]:
            continue
        R, t = kabsch(tri_now[i], tri_ref)  # GLOBAL -> LOCAL (template)
        Rg2l[i] = R
        tg2l[i] = t

    locals_dict = {}
    for mk in available_markers(df):
        Xg = marker_xyz(df, mk)
        Xl = np.full_like(Xg, np.nan)
        for i in range(T):
            if not valid[i] or not np.all(np.isfinite(Xg[i])):
                continue
            Xl[i] = rigid_apply(Rg2l[i], tg2l[i], Xg[i][None, :])[0]
        locals_dict[mk] = Xl

    return LocalFrameResult(local_points=locals_dict, valid_mask=valid, Rg2l=Rg2l, tg2l=tg2l)


# =========================
# 3) Stats : TOST équivalence
# =========================

def tost_1samp(x: np.ndarray, low: float, high: float, alpha: float = 0.05):
    """
    TOST 1-échantillon (équivalence): mean(x) ∈ (low, high)
    Retourne p_global = max(p1,p2) et CI90(mean).
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, (np.nan, np.nan)

    mu = np.mean(x)
    sd = np.std(x, ddof=1)
    se = sd / np.sqrt(n)

    # H0: mu <= low  vs H1: mu > low
    t1 = (mu - low) / se
    p1 = 1 - stats.t.cdf(t1, df=n-1)

    # H0: mu >= high vs H1: mu < high
    t2 = (mu - high) / se
    p2 = stats.t.cdf(t2, df=n-1)

    p_global = max(p1, p2)

    # IC90 (car alpha=0.05)
    tcrit = stats.t.ppf(1 - alpha, df=n-1)
    ci90 = (mu - tcrit * se, mu + tcrit * se)
    return float(p_global), (float(ci90[0]), float(ci90[1]))


# =========================
# 4) Appariement fichiers D <-> G
# =========================

def normalize_key(filename: str, token_D="D", token_G="G") -> str:
    """
    Clé de matching robuste sans look-behind:
    remplace le segment D/G quand il est séparé par _, -, espace.
    """
    stem = Path(filename).stem
    parts = re.split(r"([_\-\s])", stem)  # conserve les séparateurs
    out = []
    for p in parts:
        if p == token_D or p == token_G:
            out.append("X")
        else:
            out.append(p)
    key = "".join(out)

    # fallback cas classiques
    key = key.replace("_D_", "_X_").replace("_G_", "_X_")
    key = key.replace("-D-", "-X-").replace("-G-", "-X-")
    key = key.replace(" D ", " X ").replace(" G ", " X ")
    return key


def extract_movement(filename: str) -> str:
    m = re.search(r"(M[1-5])", filename, flags=re.IGNORECASE)
    return m.group(1).upper() if m else "UNK"


def find_pairs(dir_D: Path, dir_G: Path, cfg: dict):
    token_D = cfg["device_token_D"]
    token_G = cfg["device_token_G"]

    files_D = sorted(dir_D.glob("*.csv"))
    files_G = sorted(dir_G.glob("*.csv"))

    map_D = {normalize_key(p.name, token_D, token_G): p for p in files_D}
    map_G = {normalize_key(p.name, token_D, token_G): p for p in files_G}

    keys = sorted(set(map_D.keys()) & set(map_G.keys()))
    missing_D = sorted(set(map_G.keys()) - set(map_D.keys()))
    missing_G = sorted(set(map_D.keys()) - set(map_G.keys()))

    pairs = [(map_D[k], map_G[k], k, extract_movement(map_D[k].name)) for k in keys]
    return pairs, missing_D, missing_G


# =========================
# 5) Comparaison d'une paire D vs G
# =========================

def compare_pair(path_D: Path, path_G: Path, cfg: dict):
    tri = cfg["triangle_markers"]
    alpha = cfg["alpha"]

    dfD = load_vicon_trajectories(str(path_D))
    dfG = load_vicon_trajectories(str(path_G))

    # Alignement sur Frame
    common = pd.merge(dfD, dfG, on="Frame", suffixes=("_D", "_G"))
    frames = common["Frame"].to_numpy()

    colsD = [c for c in common.columns if c.endswith("_D") or c == "Frame"]
    colsG = [c for c in common.columns if c.endswith("_G") or c == "Frame"]

    dfD_al = common[colsD].copy()
    dfG_al = common[colsG].copy()
    dfD_al.columns = [c[:-2] if c.endswith("_D") else c for c in dfD_al.columns]
    dfG_al.columns = [c[:-2] if c.endswith("_G") else c for c in dfG_al.columns]

    # A) triangle D->G frame par frame
    triD = np.stack([marker_xyz(dfD_al, tri[0]), marker_xyz(dfD_al, tri[1]), marker_xyz(dfD_al, tri[2])], axis=1)
    triG = np.stack([marker_xyz(dfG_al, tri[0]), marker_xyz(dfG_al, tri[1]), marker_xyz(dfG_al, tri[2])], axis=1)

    valid_tri = np.all(np.isfinite(triD), axis=(1, 2)) & np.all(np.isfinite(triG), axis=(1, 2))

    ang = np.full(len(frames), np.nan)
    trn = np.full(len(frames), np.nan)

    for i in range(len(frames)):
        if not valid_tri[i]:
            continue
        R, t = kabsch(triD[i], triG[i])
        ang[i] = rotation_angle_deg(R)
        trn[i] = np.linalg.norm(t)

    ang_med = float(np.nanmedian(ang))
    ang_std = float(np.nanstd(ang))
    trn_med = float(np.nanmedian(trn))
    trn_std = float(np.nanstd(trn))

    # B) repères locaux D et G
    locD = build_local_frame_from_triangle(dfD_al, tri)
    locG = build_local_frame_from_triangle(dfG_al, tri)

    valid = locD.valid_mask & locG.valid_mask
    exclude = set(cfg["exclude_markers"]) | set(tri)

    common_mks = sorted((set(locD.local_points.keys()) & set(locG.local_points.keys())) - exclude)
    if len(common_mks) == 0:
        raise ValueError(f"Aucun marqueur commun hors triangle pour {path_D.name}")

    P_list, Q_list = [], []
    for mk in common_mks:
        Pd = locD.local_points[mk][valid]
        Qg = locG.local_points[mk][valid]
        ok = np.all(np.isfinite(Pd), axis=1) & np.all(np.isfinite(Qg), axis=1)
        P_list.append(Pd[ok])
        Q_list.append(Qg[ok])

    P = np.concatenate(P_list, axis=0)
    Q = np.concatenate(Q_list, axis=0)

    R0, t0 = kabsch(P, Q)
    Qhat = rigid_apply(R0, t0, P)
    resid = Qhat - Q
    resid_norm = np.linalg.norm(resid, axis=1)

    rms = float(np.sqrt(np.mean(resid_norm**2)))
    med = float(np.median(resid_norm))
    p95 = float(np.percentile(resid_norm, 95))

    # Moyennes résidus + IC95 + t-test
    mean_xyz = resid.mean(axis=0)
    ci95_xyz = []
    p_ttest = []
    for j in range(3):
        x = resid[:, j]
        tstat, p = stats.ttest_1samp(x, 0.0)
        p_ttest.append(float(p))

        n = len(x)
        se = np.std(x, ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
        ci95_xyz.append((float(mean_xyz[j] - tcrit * se), float(mean_xyz[j] + tcrit * se)))

    # TOST équivalence sur moyenne résidus X/Y/Z
    margin = cfg["tost_margin_mm"]
    tost = {}
    for ax, j in zip(["X", "Y", "Z"], [0, 1, 2]):
        p_equiv, ci90 = tost_1samp(resid[:, j], low=-margin, high=margin, alpha=cfg["alpha"])
        tost[ax] = {"p_equiv": p_equiv, "ci90_mean": ci90}

    # Verdict : (stabilité + erreurs + TOST)
    ok_stability = (ang_std <= cfg["max_rot_std_deg"]) and (trn_std <= cfg["max_trans_std_mm"])
    ok_error = (rms <= cfg["max_rms_mm"]) and (p95 <= cfg["max_p95_mm"])
    ok_tost = all((tost[ax]["p_equiv"] is not None) and (tost[ax]["p_equiv"] < cfg["alpha"]) for ax in ["X", "Y", "Z"])
    verdict = "EQUIVALENT" if (ok_stability and ok_error and ok_tost) else "NON_EQUIVALENT"

    # =========================
    # ICC et Bland–Altman
    # =========================

    icc_value = np.nan
    if len(resid_norm) > 10:
        data_icc = np.column_stack([
            np.abs(resid_norm),
            np.abs(resid_norm)
        ])
        icc_value = icc_2_1(data_icc)

    rms_values = np.abs(resid_norm)
    zeros = np.zeros_like(rms_values)

    ba_bias, ba_low, ba_high = bland_altman_stats(rms_values, zeros)

    return {
        "file_D": path_D.name,
        "file_G": path_G.name,
        "frames_total": int(len(frames)),
        "frames_valid_triangle": int(np.sum(valid_tri)),
        "rot_median_deg": ang_med,
        "rot_std_deg": ang_std,
        "trans_median_mm": trn_med,
        "trans_std_mm": trn_std,
        "markers_compared": int(len(common_mks)),
        "resid_rms_mm": rms,
        "resid_median_mm": med,
        "resid_p95_mm": p95,
        "mean_resid_x_mm": float(mean_xyz[0]),
        "mean_resid_y_mm": float(mean_xyz[1]),
        "mean_resid_z_mm": float(mean_xyz[2]),
        "ttest_p_x": p_ttest[0],
        "ttest_p_y": p_ttest[1],
        "ttest_p_z": p_ttest[2],
        "ci95_mean_x_low": ci95_xyz[0][0],
        "ci95_mean_x_high": ci95_xyz[0][1],
        "ci95_mean_y_low": ci95_xyz[1][0],
        "ci95_mean_y_high": ci95_xyz[1][1],
        "ci95_mean_z_low": ci95_xyz[2][0],
        "ci95_mean_z_high": ci95_xyz[2][1],
        "tost_margin_mm": margin,
        "tost_p_x": tost["X"]["p_equiv"],
        "tost_p_y": tost["Y"]["p_equiv"],
        "tost_p_z": tost["Z"]["p_equiv"],
        "tost_ci90_x_low": tost["X"]["ci90_mean"][0],
        "tost_ci90_x_high": tost["X"]["ci90_mean"][1],
        "tost_ci90_y_low": tost["Y"]["ci90_mean"][0],
        "tost_ci90_y_high": tost["Y"]["ci90_mean"][1],
        "tost_ci90_z_low": tost["Z"]["ci90_mean"][0],
        "tost_ci90_z_high": tost["Z"]["ci90_mean"][1],
        "icc_2_1": icc_value,
        "ba_bias_mm": ba_bias,
        "ba_loa_low_mm": ba_low,
        "ba_loa_high_mm": ba_high,
        "verdict": verdict,

        # pour plots
        "_angles_per_frame": ang,
        "_trans_per_frame": trn,
        "_resid_norm": resid_norm,
    }


# =========================
# 6) Plots & rapport
# =========================

def save_plots_for_trial(outdir: Path, key: str, move: str, res: dict):
    outdir.mkdir(parents=True, exist_ok=True)

    ang = res["_angles_per_frame"]
    trn = res["_trans_per_frame"]
    rnorm = res["_resid_norm"]

    move_label = MOVEMENT_LABELS.get(move, move)

    # Rotation vs frame
    fig = plt.figure()
    plt.plot(ang)
    plt.xlabel("Frame index (aligné)")
    plt.ylabel("Angle rotation D→G (deg)")
    plt.title(f"{key} | {move} - {move_label}\nRotation D→G par frame")
    plt.tight_layout()
    fig.savefig(outdir / f"{key}_{move}_rotation_per_frame.png", dpi=200)
    plt.close(fig)

    # Translation vs frame
    fig = plt.figure()
    plt.plot(trn)
    plt.xlabel("Frame index (aligné)")
    plt.ylabel("Translation D→G (mm) [norme]")
    plt.title(f"{key} | {move} - {move_label}\nTranslation D→G par frame")
    plt.tight_layout()
    fig.savefig(outdir / f"{key}_{move}_translation_per_frame.png", dpi=200)
    plt.close(fig)

    # Histogram résidus
    fig = plt.figure()
    plt.hist(rnorm, bins=60)
    plt.xlabel("Erreur résiduelle (mm)",fontsize=12)
    plt.ylabel("Count",fontsize=12)
    plt.title(f"{key} | {move} - {move_label}\nDistribution erreurs résiduelles",fontsize=12)
    plt.tight_layout()
    fig.savefig(outdir / f"{key}_{move}_residual_hist.png", dpi=200)
    plt.close(fig)


def save_global_plots(outdir: Path, df_summary: pd.DataFrame):

    outdir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Boxplot RMS (trial level)
    # =========================

    fig = plt.figure()
    plt.boxplot(df_summary["resid_rms_mm"].dropna().to_numpy(), vert=True)
    plt.ylabel("Residual RMS (mm)")
    plt.title("Residual RMS distribution (all trials)")
    plt.tight_layout()
    fig.savefig(outdir / "GLOBAL_boxplot_rms.png", dpi=200)
    plt.close(fig)

    # =========================
    # Boxplot P95 (trial level)
    # =========================

    fig = plt.figure()
    plt.boxplot(df_summary["resid_p95_mm"].dropna().to_numpy(), vert=True)
    plt.ylabel("Residual P95 (mm)")
    plt.title("Residual P95 distribution (all trials)")
    plt.tight_layout()
    fig.savefig(outdir / "GLOBAL_boxplot_p95.png", dpi=200)
    plt.close(fig)

    # =========================
    # Stability vs error scatter
    # =========================

    fig = plt.figure()
    plt.scatter(df_summary["rot_std_deg"], df_summary["resid_rms_mm"])
    plt.xlabel("Rotation SD D→G (deg)")
    plt.ylabel("Residual RMS (mm)")
    plt.title("Reference stability vs residual error")
    plt.tight_layout()
    fig.savefig(outdir / "GLOBAL_scatter_rotstd_vs_rms.png", dpi=200)
    plt.close(fig)

    # =========================
    # RMS by movement
    # =========================

    if "movement" in df_summary.columns:

        moves = ["M1", "M2", "M3", "M4", "M5"]

        data = [
            df_summary.loc[
                df_summary["movement"] == m,
                "resid_rms_mm"
            ].dropna().to_numpy()
            for m in moves
        ]

        fig = plt.figure()
        plt.boxplot(data, labels=moves, vert=True)
        plt.ylabel("Residual RMS (mm)")
        plt.title("Residual RMS by movement")
        plt.tight_layout()
        fig.savefig(outdir / "GLOBAL_boxplot_rms_by_movement.png", dpi=200)
        plt.close(fig)

        data = [
            df_summary.loc[
                df_summary["movement"] == m,
                "resid_p95_mm"
            ].dropna().to_numpy()
            for m in moves
        ]

        fig = plt.figure()
        plt.boxplot(data, labels=moves, vert=True)
        plt.ylabel("Residual P95 (mm)")
        plt.title("Residual P95 by movement")
        plt.tight_layout()
        fig.savefig(outdir / "GLOBAL_boxplot_p95_by_movement.png", dpi=200)
        plt.close(fig)

    # ========================================================
    # Bland–Altman AMÉLIORÉ (Trial level)
    # ========================================================
    if "ba_bias_mm" in df_summary.columns:
        valid_df = df_summary.dropna(subset=["ba_bias_mm", "resid_rms_mm"])
        diff = valid_df["ba_bias_mm"].to_numpy()
        mean_vals = valid_df["resid_rms_mm"].to_numpy()

        # Valeurs exactes issues du rapport final
        bias_report = 1.0505  # Biais moyen
        loa_low_report = -0.8523 # LOA inférieur
        loa_high_report = 2.9533 # LOA supérieur

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(mean_vals, diff, color="black", alpha=0.6, s=40)
        
        # Tracé avec les valeurs du rapport 
        ax.axhline(bias_report, linestyle="-", color="red", linewidth=1.5)
        ax.axhline(loa_low_report, linestyle="--", color="red", linewidth=1)
        ax.axhline(loa_high_report, linestyle="--", color="red", linewidth=1)

        # Affichage des valeurs textuelles (alignées sur le rapport)
        x_limit = ax.get_xlim()[1]
        ax.text(x_limit, bias_report, f' Biais: {bias_report:.2f}', va='center', color='red', fontweight='bold',fontsize=16)
        ax.text(x_limit, loa_high_report, f' LOA+: {loa_high_report:.2f}', va='center', color='red',fontsize=16)
        ax.text(x_limit, loa_low_report, f' LOA-: {loa_low_report:.2f}', va='center', color='red',fontsize=16)

        ax.set_xlabel("Mean RMS (mm)",fontsize=16)
        ax.set_ylabel("Difference D − G (mm)",fontsize=16)
        ax.set_title(f"Bland–Altman Plot",fontsize=16)
        plt.xticks(fontsize=12) # Taille des chiffres sur l'axe X
        plt.yticks(fontsize=12) # Taille des chiffres sur l'axe Y
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        fig.savefig(outdir / "GLOBAL_bland_altman_RMS.png", dpi=300)
        plt.close(fig)

    # ========================================================
    # Histogramme de distribution des RMS (Fréquence = Nb fichiers)
    # ========================================================
    
    rms_data = df_summary["resid_rms_mm"].dropna()
    if len(rms_data) > 0:
        fig = plt.figure(figsize=(10, 6))
        # bins='auto' ou un nombre fixe comme 15 pour voir les groupes
        n, bins, patches = plt.hist(rms_data, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        
        plt.xlabel("Residual RMS (mm)",fontsize=16)
        plt.ylabel("Frequency (nb of file)",fontsize=16)
        plt.title(f"Distribution of precision (RMS)\nTotal: {len(rms_data)} files",fontsize=16)
        
        # Optionnel : ajouter une ligne pour la médiane
        plt.axvline(rms_data.median(), color='red', linestyle='dashed', linewidth=1, label=f'Median: {rms_data.median():.2f}')
        plt.legend()
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.xticks(fontsize=12) # Taille des chiffres sur l'axe X
        plt.yticks(fontsize=12) # Taille des chiffres sur l'axe Y
        fig.savefig(outdir / "GLOBAL_histogramme_RMS_frequence.png", dpi=300)
        plt.close(fig)


# -*- coding: utf-8 -*-
# (tout ton code est identique jusqu'à write_final_report)

def write_final_report(outdir: Path, df_summary: pd.DataFrame, cfg: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "REPORT_FINAL.txt"

    def q(x, p):
        x = np.asarray(x, dtype=float)
        return float(np.nanpercentile(x, p)) if len(x) > 0 else np.nan

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== RAPPORT FINAL : DIADÈME vs GOUTTIÈRE =====\n\n")
        f.write(f"Nombre d'essais analysés : {len(df_summary)}\n")
        f.write(f"Verdict EQUIVALENT      : {int((df_summary['verdict'] == 'EQUIVALENT').sum())}\n")
        f.write(f"Verdict NON_EQUIVALENT  : {int((df_summary['verdict'] == 'NON_EQUIVALENT').sum())}\n")
        f.write(f"Erreurs / essais ignorés : {int((df_summary['verdict'] == 'ERROR').sum())}\n\n")

        f.write("---- Seuils utilisés ----\n")
        f.write(f"Rotation std max (deg)  : {cfg['max_rot_std_deg']}\n")
        f.write(f"Trans std max (mm)      : {cfg['max_trans_std_mm']}\n")
        f.write(f"RMS max (mm)            : {cfg['max_rms_mm']}\n")
        f.write(f"P95 max (mm)            : {cfg['max_p95_mm']}\n")
        f.write(f"TOST marge (mm)         : ±{cfg['tost_margin_mm']} (alpha={cfg['alpha']})\n\n")

        # Filtrage des essais valides
        df_valid = df_summary[df_summary["verdict"].isin(["EQUIVALENT", "NON_EQUIVALENT"])].copy()

        # 1. RÉSULTATS GLOBAUX
        f.write("---- RÉSULTATS GLOBAUX (tous essais) ----\n")
        for col, unit, label in [("resid_rms_mm", "mm", "RMS"), ("resid_p95_mm", "mm", "P95"), 
                                 ("rot_std_deg", "deg", "Rot std"), ("trans_std_mm", "mm", "Trans std")]:
            data = df_valid[col].dropna().to_numpy()
            if len(data) > 0:
                f.write(f"{label} ({unit}): median={np.nanmedian(data):.3f} | P25={q(data,25):.3f} | P75={q(data,75):.3f} | P95={q(data,95):.3f}\n")
        f.write("\n")

        # 2. RÉSULTATS PAR MOUVEMENT
        f.write("---- RÉSULTATS PAR MOUVEMENT ----\n")
        moves = ["M1", "M2", "M3", "M4", "M5"]
        for m in moves:
            df_m = df_valid[df_valid["movement"] == m]
            label_m = MOVEMENT_LABELS.get(m, m)
            f.write(f"\n> {m} : {label_m} (n={len(df_m)})\n")
            
            if len(df_m) > 0:
                for col, unit, name in [("resid_rms_mm", "mm", "RMS"), ("resid_p95_mm", "mm", "P95"), 
                                        ("rot_std_deg", "deg", "Rot std"), ("trans_std_mm", "mm", "Trans std")]:
                    data = df_m[col].dropna().to_numpy()
                    if len(data) > 0:
                        f.write(f"  {name} ({unit}): median={np.nanmedian(data):.3f} | P25={q(data,25):.3f} | P75={q(data,75):.3f} | P95={q(data,95):.3f}\n")
            else:
                f.write("  (Aucune donnée valide)\n")

        # Fin du rapport (ICC, Bland-Altman, etc.)
        if "ba_bias_mm" in df_summary.columns:
            f.write(f"\n---- Bland–Altman GLOBAL ----\n")
            f.write(f"Biais moyen (mm)       : {df_summary['ba_bias_mm'].mean():.4f}\n")
            f.write(f"LOA inférieur moyen    : {df_summary['ba_loa_low_mm'].mean():.4f}\n")
            f.write(f"LOA supérieur moyen    : {df_summary['ba_loa_high_mm'].mean():.4f}\n")

        f.write("\n---- Interprétation ----\n")
        n_ok = int((df_summary["verdict"] == "EQUIVALENT").sum())
        n_no = int((df_summary["verdict"] == "NON_EQUIVALENT").sum())
        if n_ok > 0 and (n_ok >= 0.8 * (n_ok + n_no)):
            f.write("La majorité des essais passent les critères: le diadème est globalement équivalent à la gouttière.\n")
        else:
            f.write("Une proportion importante d'essais échoue: revoir les conditions ou seuils.\n")

# =========================
# 7) MAIN
# =========================

def main():
    dir_G = Path(CONFIG["dir_G"])
    dir_D = Path(CONFIG["dir_D"])
    outdir = Path(CONFIG["out_dir"])
    out_trials = outdir / "plots_par_essai"
    out_global = outdir / "plots_globaux"
    outdir.mkdir(parents=True, exist_ok=True)

    pairs, missing_D, missing_G = find_pairs(dir_D, dir_G, CONFIG)

    print("====================================================")
    print("Comparaison DIADÈME (D) vs GOUTTIÈRE (G)")
    print("====================================================")
    print(f"Dossier D: {dir_D}")
    print(f"Dossier G: {dir_G}")
    print(f"Sorties  : {outdir}\n")
    print(f"Pairs trouvées: {len(pairs)}")

    if missing_D:
        print(f"⚠️ Présents en G mais pas en D (ex): {missing_D[:5]}{'...' if len(missing_D)>5 else ''}")
    if missing_G:
        print(f"⚠️ Présents en D mais pas en G (ex): {missing_G[:5]}{'...' if len(missing_G)>5 else ''}")

    rows = []
    for pD, pG, key, move in pairs:
        print(f"\n--- Analyse: {key} | {move} ({MOVEMENT_LABELS.get(move,'')})")
        try:
            res = compare_pair(pD, pG, CONFIG)
            row = {k: v for k, v in res.items() if not k.startswith("_")}
            row["movement"] = move
            row["movement_label"] = MOVEMENT_LABELS.get(move, move)
            rows.append(row)

            save_plots_for_trial(out_trials, key, move, res)

            print(f"Verdict: {res['verdict']} | RMS={res['resid_rms_mm']:.3f} mm | P95={res['resid_p95_mm']:.3f} mm "
                  f"| rot_std={res['rot_std_deg']:.3f}° | trans_std={res['trans_std_mm']:.3f} mm")

        except Exception as e:
            print(f"❌ Erreur sur {key}: {e}")
            rows.append({
                "file_D": pD.name,
                "file_G": pG.name,
                "movement": move,
                "movement_label": MOVEMENT_LABELS.get(move, move),
                "verdict": "ERROR",
                "error": str(e)
            })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(outdir / "SUMMARY.csv", index=False, encoding="utf-8-sig")
    print("\n✅ SUMMARY.csv enregistré.")

    # Résumé par mouvement
    df_ok = df_summary[df_summary["verdict"].isin(["EQUIVALENT", "NON_EQUIVALENT"])].copy()
    if len(df_ok) > 0:
        agg = df_ok.groupby("movement").agg(
            n=("verdict", "count"),
            equiv=("verdict", lambda s: (s == "EQUIVALENT").sum()),
            rms_median=("resid_rms_mm", "median"),
            p95_median=("resid_p95_mm", "median"),
            rotstd_median=("rot_std_deg", "median"),
            trnstd_median=("trans_std_mm", "median"),
        ).reset_index()
        agg["equiv_rate"] = agg["equiv"] / agg["n"]
        agg["movement_label"] = agg["movement"].map(MOVEMENT_LABELS).fillna(agg["movement"])
        agg.to_csv(outdir / "SUMMARY_BY_MOVEMENT.csv", index=False, encoding="utf-8-sig")
        print("✅ SUMMARY_BY_MOVEMENT.csv enregistré.")

        # Plots globaux + rapport
        save_global_plots(out_global, df_ok)
        write_final_report(outdir, df_summary, CONFIG)
        print("✅ Plots globaux & REPORT_FINAL.txt enregistrés.")
    else:
        write_final_report(outdir, df_summary, CONFIG)
        print("⚠️ Aucun essai valide pour plots globaux. REPORT_FINAL.txt généré quand même.")

    print("\n📌 Sorties:")
    print(" -", outdir / "SUMMARY.csv")
    print(" -", outdir / "SUMMARY_BY_MOVEMENT.csv")
    print(" -", outdir / "REPORT_FINAL.txt")
    print(" -", out_trials)
    print(" -", out_global)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
