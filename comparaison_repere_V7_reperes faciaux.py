# -*- coding: utf-8 -*-
"""
======================================================================
COMPARAISON DENTAL SPLINT vs HEADBAND — V9 (publication)
======================================================================

Nouveautés V9
-------------
1. LABELS HARMONISÉS pour publication :
   - "Dental splint" (au lieu de "Gutter")
   - "Headband"
   - "Splint" (abréviation)
   Tous les axes, titres et légendes sont nettoyés.

2. NOUVELLE FIGURE Bland-Altman sur la ROM 3D (celle de l'article).
   L'ancienne Fig01_BlandAltman_Pics.png est conservée en
   Fig01b_BlandAltman_Peaks.png pour le supplementary.

3. NOMS DE FICHIERS alignés avec le script Word :
   - Fig01_devices.png        (photo, à copier manuellement)
   - Fig02_NRMSE_by_Movement.png
   - Fig06_Example_Norms.png
   - Fig08_Static_Noise_G_vs_D.png
   - Fig10_Diadem_Drift_Series.png
   - Fig11_Stability_by_Movement.png
   - Fig12_Stability_by_Subject.png
   - Fig13_Peaks_vs_Cutoff.png
   - Fig14_Rigidity_Diagnostics.png
   - Fig16_Skin_Drift_Comparison.png
   - Fig_BlandAltman_ROM.png   (NEW)
======================================================================
"""

from pathlib import Path

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr, kruskal, friedmanchisquare, wilcoxon


# =====================================================================
# CONFIGURATION
# =====================================================================

DIR_G = Path(
    r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe"
    r"\DATA\gouttiere\tous fichiers gouttiere"
)

DIR_D = Path(
    r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe"
    r"\DATA\diademe\tous fichiers Diademe"
)

OUT = Path(
    r"C:\Users\felima\Desktop\Face MoCap\Analyse MoCap\Test Diademe"
    r"\RESULTATS\RESULTATS_COMPARAISON_V9_FIXED TRESHOLD 2.5"
)

REF_POINTS = ["M1", "M2", "M3"]

MOVING_MARKERS = {
    "M1": ["Lc2", "Lc5", "Pps01", "Pps02"],
}

# --- Sélection empirique des marqueurs fixes --------------------------
FIXED_THRESHOLD_MM = 2.5
MIN_FIXED_MARKERS = 10
MIN_SKIN_FRAMES = 30

# --- Acquisition ------------------------------------------------------
FPS_DEFAULT = 100.0

# --- Filtrage / dérivation --------------------------------------------
FILTER_CUTOFF_HZ = 6.0
FILTER_ORDER = 2
EDGE_TRIM_FRAMES = 5

SWEEP_CUTOFFS_HZ = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# --- Trous et qualité -------------------------------------------------
MAX_GAP_FRAMES = 10
MIN_RUN_PCT = 70.0
MAX_INTERP_PCT = 15.0
USE_QC_FOR_MAIN = False

# --- Validité ---------------------------------------------------------
MIN_VALID_FRAMES = 30
EPS = 1e-9

# --- Ligne de base moyennée -------------------------------------------
BASELINE_FRAMES = 5

# --- Mode de repère ---------------------------------------------------
REPERE_MODE = "kabsch"

# --- Alignement des orientations --------------------------------------
ALIGN_ORIENTATION = True

MIN_NORM = {"Pos": 1.0, "Vel": 15.0, "Acc": 400.0}

# --- Tolérances -------------------------------------------------------
TRANSLATION_TOLERANCES_MM = [2.0, 3.0]
ROTATION_TOLERANCE_DEG = 2.0

KIN = ["Pos", "Vel", "Acc"]
UNITS = {"Pos": "mm", "Vel": "mm_s", "Acc": "mm_s2"}
LABELS = {
    "Pos": "Displacement [mm]",
    "Vel": "Speed [mm/s]",
    "Acc": "Acceleration [mm/s²]",
}

# --- Labels pour publication ------------------------------------------
LABEL_GUTTER = "Dental splint"
LABEL_HEADBAND = "Headband"
LABEL_G_SHORT = "Splint"
LABEL_D_SHORT = "Headband"

DEVICE_LABEL = {
    "Gutter": LABEL_GUTTER,
    "Headband": LABEL_HEADBAND,
}

# Style matplotlib
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def col(prefix, metric):
    return f"{prefix}_{metric}_{UNITS[prefix]}"


def unit_label(p):
    return UNITS[p].replace("_", "/")


# =====================================================================
# OUTILS GÉNÉRAUX
# =====================================================================

def rotation_angle_from_matrix(R):
    value = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(value))


def mean_rotation(rotations):
    M = np.mean(rotations, axis=0)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def longest_run(mask):
    best = (0, 0)
    start = None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def fill_short_gaps(arr, max_gap):
    out = arr.copy()
    n = len(arr)
    bad = ~np.isfinite(arr).all(axis=1)
    filled = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                xs = np.arange(i, j)
                for k in range(arr.shape[1]):
                    out[i:j, k] = np.interp(
                        xs, [i - 1, j], [arr[i - 1, k], arr[j, k]])
                filled[i:j] = True
            i = j
        else:
            i += 1
    return out, filled


def lowpass(data, fs, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER):
    wn = min(cutoff / (fs / 2.0), 0.99)
    b, a = butter(order, wn)
    return filtfilt(b, a, data, axis=0)


def kinematics(xyz, fs, cutoff=FILTER_CUTOFF_HZ,
               baseline_frames=BASELINE_FRAMES):
    dt = 1.0 / fs
    filtered = lowpass(xyz, fs, cutoff)

    vel = np.gradient(filtered, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)

    t = EDGE_TRIM_FRAMES
    sl = slice(t, len(filtered) - t) if t > 0 else slice(None)

    n_base = min(baseline_frames, len(filtered))
    baseline = np.mean(filtered[:n_base], axis=0)

    pos = filtered[sl] - baseline
    return {"Pos": pos, "Vel": vel[sl], "Acc": acc[sl]}


def spread(kin):
    return {
        "Pos": float(np.sqrt(np.mean(np.var(kin["Pos"], axis=0)))),
        "Vel": float(np.sqrt(np.mean(np.sum(kin["Vel"] ** 2, axis=1)))),
        "Acc": float(np.sqrt(np.mean(np.sum(kin["Acc"] ** 2, axis=1)))),
    }


def safe_pearson(a, b):
    if np.std(a) > EPS and np.std(b) > EPS:
        try:
            return pearsonr(a, b)[0]
        except Exception:
            return np.nan
    return np.nan


def marker_class(movement, marker):
    if movement in MOVING_MARKERS:
        return "moving" if marker in MOVING_MARKERS[movement] else "static"
    return "moving"


# =====================================================================
# REPÈRE LOCAL
# =====================================================================

def get_local_basis_m1m2m3(m1, m2, m3):
    try:
        m1 = np.asarray(m1, dtype=float)
        m2 = np.asarray(m2, dtype=float)
        m3 = np.asarray(m3, dtype=float)

        if not (np.isfinite(m1).all() and np.isfinite(m2).all()
                and np.isfinite(m3).all()):
            return None, None

        vx = m2 - m1
        nx = np.linalg.norm(vx)
        if nx < EPS:
            return None, None
        vx = vx / nx

        vz = np.cross(vx, m3 - m1)
        nz = np.linalg.norm(vz)
        if nz < EPS:
            return None, None
        vz = vz / nz

        vy = np.cross(vz, vx)
        ny = np.linalg.norm(vy)
        if ny < EPS:
            return None, None
        vy = vy / ny

        return np.column_stack((vx, vy, vz)), m1.copy()

    except Exception:
        return None, None


def kabsch(P, Q, weights=None):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n = len(P)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    centroid_P = np.sum(w[:, None] * P, axis=0)
    centroid_Q = np.sum(w[:, None] * Q, axis=0)
    H = ((P - centroid_P) * w[:, None]).T @ (Q - centroid_Q)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_Q - R @ centroid_P
    Q_fit = (R @ P.T).T + t
    rmsd = np.sqrt(np.sum(w * np.sum((Q_fit - Q) ** 2, axis=1)))
    return R, t, rmsd


def build_reference_triangle(data, ref_points, n_frames, baseline_frames):
    pts = []
    for name in ref_points:
        if name not in data:
            return None
        arr = data[name]
        n = min(len(arr), n_frames)
        valid = [f for f in range(n)
                 if np.isfinite(arr[f]).all()][:baseline_frames]
        if len(valid) < 2:
            valid = [f for f in range(n) if np.isfinite(arr[f]).all()][:1]
        if not valid:
            return None
        pts.append(np.mean(arr[valid], axis=0))
    return np.array(pts)


def get_local_basis_kabsch(m1, m2, m3, ref_triangle, ref_basis):
    try:
        P = np.array([m1, m2, m3], dtype=float)
        if not np.isfinite(P).all():
            return None, None
        R_k, t_k, _ = kabsch(ref_triangle, P)
        centroid_ref = np.mean(ref_triangle, axis=0)
        R_local = R_k @ ref_basis
        origin_local = t_k + R_k @ centroid_ref
        return R_local, origin_local
    except Exception:
        return None, None


def point_to_local(point_global, R, origin):
    return R.T @ (point_global - origin)


def relative_transform_gutter_to_diadem(R_g, origin_g, R_d, origin_d):
    return R_g.T @ (origin_d - origin_g), R_g.T @ R_d


# =====================================================================
# REPÈRE CUTANÉ
# =====================================================================

def amplitude_in_frame(positions, R_frames, origin_frames, n_frames):
    if positions is None or R_frames is None or origin_frames is None:
        return np.nan

    local = []
    for f in range(n_frames):
        R = R_frames[f]
        o = origin_frames[f]
        if R is None or o is None:
            continue
        p = positions[f]
        if not np.isfinite(p).all():
            continue
        if not (np.isfinite(R).all() and np.isfinite(o).all()):
            continue
        local.append(R.T @ (p - o))

    if len(local) < MIN_SKIN_FRAMES:
        return np.nan
    arr = np.array(local)
    return float(np.median(np.ptp(arr, axis=0)))


def build_skin_reference(face_positions, fixed_markers, n_frames,
                         baseline_frames=BASELINE_FRAMES,
                         min_markers=MIN_FIXED_MARKERS):
    if len(fixed_markers) < min_markers:
        return None

    ref_clouds = []
    for m in fixed_markers:
        if m not in face_positions:
            continue
        arr = face_positions[m]
        n = min(len(arr), n_frames)
        valid = [f for f in range(n)
                 if np.isfinite(arr[f]).all()][:baseline_frames]
        if len(valid) < 2:
            continue
        ref_clouds.append(np.mean(arr[valid], axis=0))

    if len(ref_clouds) < min_markers:
        return None
    ref_cloud = np.array(ref_clouds)
    M = len(ref_cloud)

    R_skin = np.full((n_frames, 3, 3), np.nan)
    O_skin = np.full((n_frames, 3), np.nan)
    rmsd = np.full(n_frames, np.nan)

    for f in range(n_frames):
        P = np.full((M, 3), np.nan)
        ok = np.zeros(M, dtype=bool)
        for i, m in enumerate(fixed_markers):
            if m not in face_positions:
                continue
            p = face_positions[m][f]
            if np.isfinite(p).all():
                P[i] = p
                ok[i] = True

        if ok.sum() < min_markers:
            continue

        R_k, t_k, r = kabsch(ref_cloud[ok], P[ok])
        R_skin[f] = R_k
        centroid_ref = np.mean(ref_cloud[ok], axis=0)
        O_skin[f] = t_k + R_k @ centroid_ref
        rmsd[f] = r

    return {
        "R_skin": R_skin,
        "O_skin": O_skin,
        "rmsd": rmsd,
        "n_markers_used": M,
        "ref_cloud": ref_cloud,
        "fixed_markers": fixed_markers,
    }


def drift_in_frame(R_target, O_target, R_ref, O_ref):
    n = len(R_target)
    trans = np.full((n, 3), np.nan)
    rot = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

    for f in range(n):
        Rt = R_target[f]
        Ot = O_target[f]
        Rr = R_ref[f]
        Or = O_ref[f]

        if Rt is None or Ot is None or Rr is None or Or is None:
            continue
        if not (np.isfinite(Rt).all() and np.isfinite(Ot).all()
                and np.isfinite(Rr).all() and np.isfinite(Or).all()):
            continue

        t = Rr.T @ (Ot - Or)
        r = Rr.T @ Rt
        trans[f] = t
        rot[f] = rotation_angle_from_matrix(r)
        ok[f] = True

    if ok.sum() < 2:
        return trans, rot, ok

    idx = np.where(ok)[0]
    first = idx[0]
    trans = trans - trans[first]
    rot = np.abs(rot - rot[first])
    return trans, rot, ok


# =====================================================================
# LECTURE VICON
# =====================================================================

def load_vicon(file_path):
    try:
        dataframe = pandas.read_csv(file_path, skiprows=4, low_memory=False)
        dataframe = dataframe.apply(pandas.to_numeric, errors="coerce")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if len(lines) < 3:
            return None, None

        try:
            fps = float(lines[1].strip().split(",")[0])
            if not np.isfinite(fps) or fps <= 0:
                fps = FPS_DEFAULT
        except Exception:
            fps = FPS_DEFAULT

        marker_names = [
            e.split(":")[-1].strip()
            for e in lines[2].strip().split(",")
            if e.strip()
        ]

        data = {}
        for i, name in enumerate(marker_names):
            if not name:
                continue
            start = i * 3 + 2
            end = start + 3
            if end > dataframe.shape[1]:
                continue
            xyz = dataframe.iloc[:, start:end].values
            if xyz.shape[1] != 3:
                continue
            data[name] = xyz.astype(float)

        if not data:
            return None, None

        return data, fps

    except Exception as error:
        print(f"[ERREUR LECTURE] {Path(file_path).name} : {error}")
        return None, None


# =====================================================================
# NOMS DE FICHIERS
# =====================================================================

def find_matching_diadem_file(gutter_file):
    name = gutter_file.name
    variants = [
        name.replace("_G_", "_D_"),
        name.replace("_g_", "_d_"),
        name.replace("G_", "D_"),
        name.replace("_G", "_D"),
    ]
    for v in variants:
        candidate = DIR_D / v
        if candidate.exists():
            return candidate
    return None


def extract_movement(file_name):
    name = Path(file_name).stem
    if "_M" not in name:
        return "M?"
    after = name.split("_M", 1)[1]
    number = ""
    for ch in after:
        if ch.isdigit():
            number += ch
        else:
            break
    return f"M{number}" if number else "M?"


def extract_subject(file_name):
    try:
        parts = Path(file_name).stem.split("_")
        return parts[2] if len(parts) >= 3 else Path(file_name).stem
    except Exception:
        return str(file_name)


# =====================================================================
# DIAGNOSTIC DE RIGIDITÉ
# =====================================================================

def rigidity_diagnostics(data, fs, file_name, subject, movement):
    if not all(m in data for m in REF_POINTS):
        return None

    m1 = data["M1"]
    m2 = data["M2"]
    m3 = data["M3"]
    n = min(len(m1), len(m2), len(m3))

    valid = (np.isfinite(m1[:n]).all(axis=1)
             & np.isfinite(m2[:n]).all(axis=1)
             & np.isfinite(m3[:n]).all(axis=1))
    if valid.sum() < 5:
        return None

    P = np.stack([m1[:n], m2[:n], m3[:n]], axis=1)[valid]

    d12 = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    d13 = np.linalg.norm(P[:, 0] - P[:, 2], axis=1)
    d23 = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)

    tri = np.mean(P, axis=0)
    a = np.linalg.norm(tri[0] - tri[1])
    b = np.linalg.norm(tri[0] - tri[2])
    c = np.linalg.norm(tri[1] - tri[2])
    base = tri[1] - tri[0]
    base_len = np.linalg.norm(base)
    if base_len > EPS:
        h = np.linalg.norm(np.cross(base, tri[2] - tri[0])) / base_len
    else:
        h = np.nan
    area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))

    n_base = min(BASELINE_FRAMES, len(P))
    tri_ref = np.mean(P[:n_base], axis=0)
    rmsd_list = []
    for p in P:
        _, _, r = kabsch(tri_ref, p)
        rmsd_list.append(r)
    rmsd = np.array(rmsd_list)

    return {
        "File": file_name,
        "Subject": subject,
        "Movement": movement,
        "N_frames": int(valid.sum()),
        "d12_mean_mm": float(np.mean(d12)),
        "d12_std_mm": float(np.std(d12)),
        "d13_mean_mm": float(np.mean(d13)),
        "d13_std_mm": float(np.std(d13)),
        "d23_mean_mm": float(np.mean(d23)),
        "d23_std_mm": float(np.std(d23)),
        "Side_a_mm": float(a),
        "Side_b_mm": float(b),
        "Side_c_mm": float(c),
        "Height_mm": float(h),
        "Area_mm2": float(area),
        "Kabsch_rmsd_mean_mm": float(np.mean(rmsd)),
        "Kabsch_rmsd_std_mm": float(np.std(rmsd)),
        "Kabsch_rmsd_max_mm": float(np.max(rmsd)),
    }


# =====================================================================
# COMPARAISON D'UNE GRANDEUR
# =====================================================================

def compare_vectors(G, D, prefix, min_norm):
    u = UNITS[prefix]

    err = D - G
    err_norm = np.linalg.norm(err, axis=1)
    nG = np.linalg.norm(G, axis=1)
    nD = np.linalg.norm(D, axis=1)

    out = {}

    rmse = np.sqrt(np.mean(err_norm ** 2))
    out[f"{prefix}_MAE_{u}"] = np.mean(err_norm)
    out[f"{prefix}_RMSE_{u}"] = rmse
    out[f"{prefix}_Max_err_{u}"] = np.max(err_norm)

    for i, axis_name in enumerate("XYZ"):
        out[f"{prefix}_Bias_{axis_name}_{u}"] = np.mean(err[:, i])
        out[f"{prefix}_RMSE_{axis_name}_{u}"] = np.sqrt(np.mean(err[:, i] ** 2))

    dnorm = nD - nG
    rmse_norm = np.sqrt(np.mean(dnorm ** 2))
    out[f"{prefix}_Bias_norm_{u}"] = np.mean(dnorm)
    out[f"{prefix}_RMSE_norm_{u}"] = rmse_norm

    peak_g = np.max(nG)
    peak_d = np.max(nD)
    out[f"{prefix}_Peak_G_{u}"] = peak_g
    out[f"{prefix}_Peak_D_{u}"] = peak_d
    out[f"{prefix}_Peak_err_{u}"] = peak_d - peak_g
    out[f"{prefix}_Peak_relerr_pct"] = (
        100.0 * (peak_d - peak_g) / peak_g if peak_g > EPS else np.nan)

    out[f"{prefix}_NRMSE_pct"] = (
        100.0 * rmse / peak_g if peak_g > EPS else np.nan)
    out[f"{prefix}_NRMSE_norm_pct"] = (
        100.0 * rmse_norm / peak_g if peak_g > EPS else np.nan)

    out[f"{prefix}_Pearson_norm"] = safe_pearson(nG, nD)

    # --- V9 : Range of Motion 3D pour Bland-Altman ROM ---
    # ROM = peak-to-peak de la norme du déplacement, sur la fenêtre analysée
    out[f"{prefix}_ROM_G_{u}"] = float(np.ptp(nG))
    out[f"{prefix}_ROM_D_{u}"] = float(np.ptp(nD))

    valid = (nG > min_norm) & (nD > min_norm)
    if valid.sum() >= 2:
        cos = np.sum(G[valid] * D[valid], axis=1) / (nG[valid] * nD[valid])
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        out[f"{prefix}_Angle_mean_deg"] = np.mean(ang)
        out[f"{prefix}_Angle_median_deg"] = np.median(ang)
        out[f"{prefix}_Angle_max_deg"] = np.max(ang)
        out[f"{prefix}_Angle_valid_pct"] = 100.0 * valid.mean()
    else:
        out[f"{prefix}_Angle_mean_deg"] = np.nan
        out[f"{prefix}_Angle_median_deg"] = np.nan
        out[f"{prefix}_Angle_max_deg"] = np.nan
        out[f"{prefix}_Angle_valid_pct"] = 0.0

    return out


# =====================================================================
# ANALYSE D'UNE PAIRE DE FICHIERS
# =====================================================================

def analyze_file_pair(gutter_data, diadem_data, file_name, fs,
                      series_store, stab_series):
    movement = extract_movement(file_name)
    subject = extract_subject(file_name)

    marker_results, stability_results = [], []
    sweep_results, skipped = [], []
    rigidity_results = []
    skin_results, skin_fixed_selection = [], []

    common = [m for m in gutter_data if m in diadem_data]
    facial = [m for m in common if m not in REF_POINTS]

    if not common:
        return (marker_results, stability_results, sweep_results, skipped,
                rigidity_results, skin_results, skin_fixed_selection)

    n_frames = min(
        min(len(gutter_data[m]), len(diadem_data[m])) for m in common
    )

    # --- Rigidité ---
    rig_g = rigidity_diagnostics(gutter_data, fs, file_name, subject, movement)
    rig_d = rigidity_diagnostics(diadem_data, fs, file_name, subject, movement)
    if rig_g is not None:
        rig_g["Device"] = "Gutter"
        rigidity_results.append(rig_g)
    if rig_d is not None:
        rig_d["Device"] = "Headband"
        rigidity_results.append(rig_d)

    # --- Repères frame par frame ---
    if REPERE_MODE == "kabsch":
        ref_tri_g = build_reference_triangle(
            gutter_data, REF_POINTS, n_frames, BASELINE_FRAMES)
        ref_tri_d = build_reference_triangle(
            diadem_data, REF_POINTS, n_frames, BASELINE_FRAMES)
        if ref_tri_g is None or ref_tri_d is None:
            return (marker_results, stability_results, sweep_results, skipped,
                    rigidity_results, skin_results, skin_fixed_selection)
        ref_basis_g, _ = get_local_basis_m1m2m3(*ref_tri_g)
        ref_basis_d, _ = get_local_basis_m1m2m3(*ref_tri_d)
        if ref_basis_g is None or ref_basis_d is None:
            return (marker_results, stability_results, sweep_results, skipped,
                    rigidity_results, skin_results, skin_fixed_selection)

    g_frames, d_frames = [], []
    for f in range(n_frames):
        if REPERE_MODE == "kabsch":
            Rg, og = get_local_basis_kabsch(
                gutter_data["M1"][f], gutter_data["M2"][f],
                gutter_data["M3"][f], ref_tri_g, ref_basis_g)
            Rd, od = get_local_basis_kabsch(
                diadem_data["M1"][f], diadem_data["M2"][f],
                diadem_data["M3"][f], ref_tri_d, ref_basis_d)
        else:
            Rg, og = get_local_basis_m1m2m3(
                gutter_data["M1"][f], gutter_data["M2"][f],
                gutter_data["M3"][f])
            Rd, od = get_local_basis_m1m2m3(
                diadem_data["M1"][f], diadem_data["M2"][f],
                diadem_data["M3"][f])
        if Rg is None or Rd is None:
            g_frames.append(None)
            d_frames.append(None)
        else:
            g_frames.append((Rg, og))
            d_frames.append((Rd, od))

    Rg_list = [gf[0] if gf is not None else None for gf in g_frames]
    Og_list = [gf[1] if gf is not None else None for gf in g_frames]
    Rd_list = [df[0] if df is not None else None for df in d_frames]
    Od_list = [df[1] if df is not None else None for df in d_frames]

    # --- Sélection empirique des marqueurs fixes ---
    fixed_markers = []
    for m in facial:
        amp = amplitude_in_frame(gutter_data[m], Rg_list, Og_list, n_frames)
        if np.isfinite(amp) and amp < FIXED_THRESHOLD_MM:
            fixed_markers.append(m)

    skin_fixed_selection.append({
        "File": file_name,
        "Subject": subject,
        "Movement": movement,
        "N_fixed_markers": len(fixed_markers),
        "Fixed_markers": "|".join(fixed_markers) if fixed_markers else "",
        "Threshold_mm": FIXED_THRESHOLD_MM,
    })

    # --- Repère cutané ---
    skin_ref = None
    if len(fixed_markers) >= MIN_FIXED_MARKERS:
        skin_ref = build_skin_reference(
            gutter_data, fixed_markers, n_frames,
            baseline_frames=BASELINE_FRAMES,
            min_markers=MIN_FIXED_MARKERS)

    # --- Pose relative diadème / splint ---
    tr_full = np.full((n_frames, 3), np.nan)
    ro_full = np.full((n_frames, 3, 3), np.nan)
    ok = np.zeros(n_frames, dtype=bool)

    for f in range(n_frames):
        if g_frames[f] is None or d_frames[f] is None:
            continue
        t, r = relative_transform_gutter_to_diadem(
            g_frames[f][0], g_frames[f][1], d_frames[f][0], d_frames[f][1])
        tr_full[f] = t
        ro_full[f] = r
        ok[f] = True

    if ok.sum() < MIN_VALID_FRAMES:
        return (marker_results, stability_results, sweep_results, skipped,
                rigidity_results, skin_results, skin_fixed_selection)

    R_align = mean_rotation(ro_full[ok])
    if not ALIGN_ORIENTATION:
        R_align = np.eye(3)

    tr = tr_full[ok]
    ro = ro_full[ok]

    drift = tr - tr[0]
    drift_norm = np.linalg.norm(drift, axis=1)
    rot_drift = np.array([
        rotation_angle_from_matrix(ro[0].T @ r) for r in ro])

    R_mean = mean_rotation(ro)
    frame_offset = rotation_angle_from_matrix(R_mean)
    rot_dev = np.array([
        rotation_angle_from_matrix(R_mean.T @ r) for r in ro])

    a, b = longest_run(ok)
    drift_speed_rms = np.nan
    if b - a >= max(MIN_VALID_FRAMES, 2 * EDGE_TRIM_FRAMES + 10):
        seg = tr_full[a:b] - tr_full[a]
        seg_f = lowpass(seg, fs)
        sp = np.linalg.norm(np.gradient(seg_f, 1.0 / fs, axis=0), axis=1)
        sl = slice(EDGE_TRIM_FRAMES, len(sp) - EDGE_TRIM_FRAMES)
        drift_speed_rms = np.sqrt(np.mean(sp[sl] ** 2))

    stability_results.append({
        "File": file_name,
        "Subject": subject,
        "Movement": movement,
        "N_frames": len(tr),

        "Trans_X_mean_mm": np.mean(drift[:, 0]),
        "Trans_Y_mean_mm": np.mean(drift[:, 1]),
        "Trans_Z_mean_mm": np.mean(drift[:, 2]),
        "Trans_X_PTP_mm": np.ptp(drift[:, 0]),
        "Trans_Y_PTP_mm": np.ptp(drift[:, 1]),
        "Trans_Z_PTP_mm": np.ptp(drift[:, 2]),

        "Trans_norm_mean_mm": np.mean(drift_norm),
        "Trans_norm_std_mm": np.std(drift_norm),
        "Trans_norm_max_drift_mm": np.max(drift_norm),
        "Trans_drift_speed_RMS_mm_s": drift_speed_rms,

        "Rot_drift_mean_deg": np.mean(rot_drift),
        "Rot_drift_std_deg": np.std(rot_drift),
        "Rot_drift_max_deg": np.max(rot_drift),

        "Frame_offset_deg": frame_offset,
        "Rot_dev_rms_deg": np.sqrt(np.mean(rot_dev ** 2)),
        "Rot_dev_max_deg": np.max(rot_dev),
    })

    stab_series[file_name] = {
        "fs": fs,
        "Trans_norm": drift_norm,
        "Trans_X": drift[:, 0],
        "Trans_Y": drift[:, 1],
        "Trans_Z": drift[:, 2],
        "Rot": rot_drift,
    }

    # =================================================================
    # DÉRIVE CROISÉE DANS LE REPÈRE CUTANÉ
    # =================================================================

    if skin_ref is not None:
        R_skin = skin_ref["R_skin"]
        O_skin = skin_ref["O_skin"]

        trans_d, rot_d, ok_d = drift_in_frame(
            Rd_list, Od_list, R_skin, O_skin)
        trans_g, rot_g, ok_g = drift_in_frame(
            Rg_list, Og_list, R_skin, O_skin)

        ok_common = ok_d & ok_g
        if ok_common.sum() >= MIN_VALID_FRAMES:
            trans_d_norm = np.linalg.norm(trans_d, axis=1)
            trans_g_norm = np.linalg.norm(trans_g, axis=1)
            rmsd_skin = skin_ref["rmsd"]

            skin_results.append({
                "File": file_name,
                "Subject": subject,
                "Movement": movement,
                "N_frames_common": int(ok_common.sum()),
                "N_fixed_markers": skin_ref["n_markers_used"],
                "Skin_RMSD_mean_mm": float(np.nanmean(rmsd_skin[ok_common])),
                "Skin_RMSD_max_mm": float(np.nanmax(rmsd_skin[ok_common])),
                "D_Trans_norm_mean_mm": float(
                    np.nanmean(trans_d_norm[ok_common])),
                "D_Trans_norm_max_mm": float(
                    np.nanmax(trans_d_norm[ok_common])),
                "D_Rot_mean_deg": float(np.nanmean(rot_d[ok_common])),
                "D_Rot_max_deg": float(np.nanmax(rot_d[ok_common])),
                "G_Trans_norm_mean_mm": float(
                    np.nanmean(trans_g_norm[ok_common])),
                "G_Trans_norm_max_mm": float(
                    np.nanmax(trans_g_norm[ok_common])),
                "G_Rot_mean_deg": float(np.nanmean(rot_g[ok_common])),
                "G_Rot_max_deg": float(np.nanmax(rot_g[ok_common])),
                "Trans_ratio_D_over_G": (
                    float(np.nanmax(trans_d_norm[ok_common])
                          / np.nanmax(trans_g_norm[ok_common]))
                    if np.nanmax(trans_g_norm[ok_common]) > EPS else np.nan),
                "Rot_ratio_D_over_G": (
                    float(np.nanmax(rot_d[ok_common])
                          / np.nanmax(rot_g[ok_common]))
                    if np.nanmax(rot_g[ok_common]) > EPS else np.nan),
            })

    # =================================================================
    # MARQUEURS FACIAUX
    # =================================================================

    min_len = max(MIN_VALID_FRAMES, 2 * EDGE_TRIM_FRAMES + 10)

    for marker in facial:

        pg = np.full((n_frames, 3), np.nan)
        pd_ = np.full((n_frames, 3), np.nan)

        for f in range(n_frames):
            if g_frames[f] is None or d_frames[f] is None:
                continue
            pg[f] = point_to_local(gutter_data[marker][f], *g_frames[f])
            pd_[f] = point_to_local(diadem_data[marker][f], *d_frames[f])

        raw_ok = np.isfinite(pg).all(axis=1) & np.isfinite(pd_).all(axis=1)
        valid_raw_pct = 100.0 * raw_ok.sum() / n_frames

        pg_f, fil_g = fill_short_gaps(pg, MAX_GAP_FRAMES)
        pd_f, fil_d = fill_short_gaps(pd_, MAX_GAP_FRAMES)

        mask = np.isfinite(pg_f).all(axis=1) & np.isfinite(pd_f).all(axis=1)
        a, b = longest_run(mask)
        run_len = b - a

        if run_len < min_len:
            skipped.append({
                "File": file_name,
                "Subject": subject,
                "Movement": movement,
                "Marker": marker,
                "Marker_class": marker_class(movement, marker),
                "Reason": f"séquence continue < {min_len} frames",
                "Valid_raw_pct": valid_raw_pct,
                "Longest_run_frames": run_len,
            })
            continue

        interp_pct = 100.0 * (fil_g | fil_d)[a:b].mean()
        run_pct = 100.0 * run_len / n_frames
        qc = "ok" if (run_pct >= MIN_RUN_PCT
                      and interp_pct <= MAX_INTERP_PCT) else "check"

        kin_g = kinematics(pg_f[a:b], fs)
        kin_d = kinematics(pd_f[a:b], fs)

        D_al = {p: kin_d[p] @ R_align.T for p in KIN}

        spr_g = spread(kin_g)
        spr_d = spread(kin_d)

        row = {
            "File": file_name,
            "Subject": subject,
            "Movement": movement,
            "Marker": marker,
            "Marker_class": marker_class(movement, marker),
            "QC": qc,
            "Fs_Hz": fs,
            "N_frames_run": run_len,
            "N_frames_analysed": len(kin_g["Pos"]),
            "Valid_raw_pct": valid_raw_pct,
            "Run_pct": run_pct,
            "Interp_pct": interp_pct,

            "G_Pos_SD_mm": spr_g["Pos"],
            "D_Pos_SD_mm": spr_d["Pos"],
            "G_Vel_RMS_mm_s": spr_g["Vel"],
            "D_Vel_RMS_mm_s": spr_d["Vel"],
            "G_Acc_RMS_mm_s2": spr_g["Acc"],
            "D_Acc_RMS_mm_s2": spr_d["Acc"],
        }

        for p in KIN:
            row.update(compare_vectors(kin_g[p], D_al[p], p, MIN_NORM[p]))
            raw = compare_vectors(kin_g[p], kin_d[p], p, MIN_NORM[p])
            row[col(p, "RMSE_unaligned")] = raw[col(p, "RMSE")]
            row[f"{p}_Angle_mean_unaligned_deg"] = raw[f"{p}_Angle_mean_deg"]

        marker_results.append(row)

        series_store[(file_name, marker)] = {
            "fs": fs,
            **{p: (kin_g[p], D_al[p]) for p in KIN},
        }

        for fc in SWEEP_CUTOFFS_HZ:
            kg = kinematics(pg_f[a:b], fs, fc)
            kd = kinematics(pd_f[a:b], fs, fc)
            for p in KIN:
                nG = np.linalg.norm(kg[p], axis=1)
                nD = np.linalg.norm(kd[p], axis=1)
                peak_g = np.max(nG)
                peak_d = np.max(nD)
                rmse_n = np.sqrt(np.mean((nD - nG) ** 2))
                sweep_results.append({
                    "File": file_name,
                    "Subject": subject,
                    "Movement": movement,
                    "Marker": marker,
                    "Marker_class": marker_class(movement, marker),
                    "QC": qc,
                    "Cutoff_Hz": fc,
                    "Quantity": p,
                    "RMSE_norm": rmse_n,
                    "Pearson_norm": safe_pearson(nG, nD),
                    "Peak_G": peak_g,
                    "Peak_D": peak_d,
                    "Peak_relerr_pct": (
                        100.0 * (peak_d - peak_g) / peak_g
                        if peak_g > EPS else np.nan),
                    "NRMSE_norm_pct": (
                        100.0 * rmse_n / peak_g if peak_g > EPS else np.nan),
                })

    return (marker_results, stability_results, sweep_results, skipped,
            rigidity_results, skin_results, skin_fixed_selection)


# =====================================================================
# SOUS-ENSEMBLES
# =====================================================================

def main_subset(df):
    if df.empty or "Marker_class" not in df.columns:
        return df
    d = df[df["Marker_class"] == "moving"]
    if USE_QC_FOR_MAIN:
        d = d[d["QC"] == "ok"]
    return d


def med_iqr(s):
    s = pandas.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return "n/a"
    return f"{s.median():.2f} [{s.quantile(.25):.2f}–{s.quantile(.75):.2f}]"


def aggregate_per_file(df, value_cols, group_cols=("File", "Subject",
                                                    "Movement")):
    if df.empty:
        return df
    avail = [c for c in value_cols if c in df.columns]
    if not avail:
        return pandas.DataFrame()
    agg = df.groupby(list(group_cols))[avail].median().reset_index()
    return agg


# =====================================================================
# BLAND–ALTMAN
# =====================================================================

def bland_altman(df, col_g, col_d):
    d = df[["Subject", "File", "Marker", col_g, col_d]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    if len(d) < 10 or d["Subject"].nunique() < 2:
        return None, None

    d["Mean"] = (d[col_g] + d[col_d]) / 2.0
    d["Diff"] = d[col_g] - d[col_d]

    method = "mixed-effects"
    try:
        import statsmodels.formula.api as smf

        model = smf.mixedlm(
            "Diff ~ 1",
            data=d,
            groups=d["Subject"],
            re_formula="1",
            vc_formula={"Trial": "0 + C(File)"},
        )
        res = model.fit(reml=True, method=["powell", "lbfgs"],
                        maxiter=2000, disp=False)
        if not res.converged:
            raise RuntimeError("non convergence")

        bias = float(res.fe_params["Intercept"])
        var_total = (float(res.cov_re.iloc[0, 0])
                     + float(np.sum(res.vcomp))
                     + float(res.scale))
        sd = np.sqrt(max(0.0, var_total))

    except Exception as error:
        method = f"classique (repli : {type(error).__name__})"
        bias = float(d["Diff"].mean())
        sd = float(d["Diff"].std(ddof=1))

    return d, {
        "Method": method,
        "N_obs": len(d),
        "N_subjects": d["Subject"].nunique(),
        "Bias": bias,
        "SD_total": sd,
        "LoA_lower": bias - 1.96 * sd,
        "LoA_upper": bias + 1.96 * sd,
    }


# =====================================================================
# FIGURES — LABELS PUBLICATION
# =====================================================================

def fig_bland_altman_rom(df_main):
    """
    V9 : Bland-Altman sur la ROM 3D (Pos), l'équivalent de la Figure 7
    de l'ancien manuscrit (bias -0.01 mm, LoA [-1.14, +1.13]).
    C'est la figure principale du manuscrit.
    """
    p = "Pos"
    cg = col(p, "ROM_G")
    cd = col(p, "ROM_D")

    if cg not in df_main.columns or cd not in df_main.columns:
        print(f"[WARN] Colonnes {cg} / {cd} absentes — figure ROM ignorée")
        return

    d = df_main[["Subject", "File", "Marker", cg, cd]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        print(f"[WARN] Pas assez de points pour Bland-Altman ROM")
        return

    d["Mean"] = (d[cg] + d[cd]) / 2.0
    d["Diff"] = d[cg] - d[cd]
    bias = float(d["Diff"].mean())
    sd = float(d["Diff"].std(ddof=1))
    loa_up = bias + 1.96 * sd
    loa_dn = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(d["Mean"], d["Diff"], alpha=0.5, s=20,
               color="steelblue", edgecolor="black", linewidth=0.3)

    ax.axhline(bias, color="black", linewidth=2,
               label=f"Bias = {bias:.2f} mm")
    ax.axhline(loa_up, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"+1.96 SD = {loa_up:.2f} mm")
    ax.axhline(loa_dn, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"−1.96 SD = {loa_dn:.2f} mm")

    ax.set_xlabel("Mean ROM 3D [mm]")
    ax.set_ylabel(f"Difference {LABEL_GUTTER} − {LABEL_HEADBAND} [mm]")
    ax.set_title("Bland-Altman agreement on 3D range of motion")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "Fig_BlandAltman_ROM.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    # Sauvegarder les stats
    pandas.DataFrame([{
        "Quantity": "ROM_3D",
        "N_obs": len(d),
        "N_subjects": d["Subject"].nunique(),
        "Bias": bias,
        "SD": sd,
        "LoA_lower": loa_dn,
        "LoA_upper": loa_up,
    }]).to_csv(OUT / "Fig_BlandAltman_ROM_stats.csv",
               index=False, encoding="utf-8-sig")
    print(f"   ✓ Fig_BlandAltman_ROM.png "
          f"(bias={bias:.2f} mm, LoA=[{loa_dn:.2f}, {loa_up:.2f}])")


def fig_bland_altman_peaks(df_main):
    """
    Bland-Altman sur les pics (supplementary S1).
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ba_rows = []

    for ax, p in zip(axes, KIN):
        d, res = bland_altman(df_main, col(p, "Peak_G"), col(p, "Peak_D"))
        if d is None:
            ax.set_title(f"{p} : données insuffisantes")
            continue

        ax.scatter(d["Mean"], d["Diff"], alpha=0.5, s=20)
        ax.axhline(res["Bias"], linewidth=2,
                   label=f"Bias = {res['Bias']:.2f}")
        ax.axhline(res["LoA_upper"], linestyle="--",
                   label=f"Upper LoA = {res['LoA_upper']:.2f}")
        ax.axhline(res["LoA_lower"], linestyle="--",
                   label=f"Lower LoA = {res['LoA_lower']:.2f}")
        ax.set_xlabel(f"Mean peak — {LABELS[p]}")
        ax.set_ylabel(f"{LABEL_GUTTER} − {LABEL_HEADBAND}")
        ax.set_title(f"{p} ({res['Method']})", fontsize=10)
        ax.legend(fontsize=8)

        ba_rows.append({"Quantity": p, **res})

    fig.suptitle("Bland–Altman on peak norm (moving markers) "
                 "— supplementary")
    plt.tight_layout()
    plt.savefig(OUT / "Fig01b_BlandAltman_Peaks.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    if ba_rows:
        pandas.DataFrame(ba_rows).to_csv(
            OUT / "05_BlandAltman.csv", index=False, encoding="utf-8-sig")


def fig_nrmse_by_movement(df_main):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, p in zip(axes, KIN):
        c = f"{p}_NRMSE_pct"
        sns.boxplot(data=df_main, x="Movement", y=c, ax=ax,
                    color="lightsteelblue")
        sns.stripplot(data=df_main, x="Movement", y=c, color="black",
                      alpha=0.35, size=3, ax=ax)
        ax.set_ylabel("NRMSE = vector RMSE / splint peak [%]")
        ax.set_title(f"{p} — normalised error")
    plt.tight_layout()
    plt.savefig(OUT / "Fig02_NRMSE_by_Movement.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_example(df_main, series_store):
    row = None
    for mv, mk in [("M1", "Lc2"), ("M5", "C02")]:
        sel = df_main[(df_main["Movement"] == mv) & (df_main["Marker"] == mk)]
        if not sel.empty:
            row = sel.iloc[0]
            break
    if row is None:
        row = df_main.iloc[0]

    s = series_store[(row["File"], row["Marker"])]
    t = np.arange(len(s["Pos"][0])) / s["fs"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, p in zip(axes, KIN):
        G, D = s[p]
        ax.plot(t, np.linalg.norm(G, axis=1), linewidth=2,
                label=LABEL_GUTTER, color="tab:blue")
        ax.plot(t, np.linalg.norm(D, axis=1), "--", linewidth=2,
                label=LABEL_HEADBAND, color="tab:orange")
        ax.set_ylabel(LABELS[p])
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        f"Representative marker — {row['Movement']} — {row['Marker']} — "
        f"{row['File']}")
    plt.tight_layout()
    plt.savefig(OUT / "Fig06_Example_Norms.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_static_noise(df_static):
    items = [
        ("G_Pos_SD_mm", "D_Pos_SD_mm", "Position SD [mm]"),
        ("G_Vel_RMS_mm_s", "D_Vel_RMS_mm_s", "Velocity RMS [mm/s]"),
        ("G_Acc_RMS_mm_s2", "D_Acc_RMS_mm_s2", "Acceleration RMS [mm/s²]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    n = len(df_static)
    for ax, (cg, cd, lab) in zip(axes, items):
        long = pandas.DataFrame({
            "Device": [LABEL_GUTTER] * n + [LABEL_HEADBAND] * n,
            "Value": np.r_[df_static[cg].values, df_static[cd].values],
        })
        sns.boxplot(data=long, x="Device", y="Value", ax=ax,
                    order=[LABEL_GUTTER, LABEL_HEADBAND])
        sns.stripplot(data=long, x="Device", y="Value", color="black",
                      alpha=0.25, size=2.5, ax=ax,
                      order=[LABEL_GUTTER, LABEL_HEADBAND])
        ax.set_ylabel(lab)
    fig.suptitle("Noise floor on static markers (no real movement)")
    plt.tight_layout()
    plt.savefig(OUT / "Fig08_Static_Noise_G_vs_D.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_cutoff_sweep(df_sweep_main):
    """
    V9 : génère directement la figure 'pics absolus vs cutoff'
    attendue par le script Word (Fig13_Peaks_vs_Cutoff.png).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, p in zip(axes, KIN):
        d = df_sweep_main[df_sweep_main["Quantity"] == p]
        if d.empty:
            continue
        gG = d.groupby("Cutoff_Hz")["Peak_G"]
        gD = d.groupby("Cutoff_Hz")["Peak_D"]
        medG, medD = gG.median(), gD.median()
        q1G, q3G = gG.quantile(.25), gG.quantile(.75)
        q1D, q3D = gD.quantile(.25), gD.quantile(.75)
        ax.plot(medG.index, medG.values, "o-", linewidth=2,
                label=LABEL_GUTTER, color="tab:blue")
        ax.fill_between(medG.index, q1G.values, q3G.values,
                        alpha=0.2, color="tab:blue")
        ax.plot(medD.index, medD.values, "s--", linewidth=2,
                label=LABEL_HEADBAND, color="tab:orange")
        ax.fill_between(medD.index, q1D.values, q3D.values,
                        alpha=0.2, color="tab:orange")
        ax.set_xlabel("Low-pass cutoff [Hz]")
        ax.set_ylabel(f"Peak [{unit_label(p)}]")
        ax.set_title(f"{p} — absolute peak vs cutoff")
        ax.legend()
    fig.suptitle("Absolute peaks across cutoffs "
                 "(stabilisation = no real-signal loss)")
    plt.tight_layout()
    plt.savefig(OUT / "Fig13_Peaks_vs_Cutoff.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_stability(df_stability, stab_series):
    # Fig10 : séries temporelles
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for s in stab_series.values():
        tt = np.arange(len(s["Trans_norm"])) / s["fs"]
        axes[0].plot(tt, s["Trans_norm"], alpha=0.5)
        axes[1].plot(tt, s["Rot"], alpha=0.5)
    for tol in TRANSLATION_TOLERANCES_MM:
        axes[0].axhline(tol, color="red", linestyle="--", linewidth=1,
                        label=f"{tol} mm")
    axes[1].axhline(ROTATION_TOLERANCE_DEG, color="red", linestyle="--",
                    linewidth=1,
                    label=f"{ROTATION_TOLERANCE_DEG}°")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Translation drift [mm]")
    axes[0].set_title(f"{LABEL_HEADBAND} translation drift "
                      f"in {LABEL_G_SHORT.lower()} frame")
    axes[0].legend()
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Rotation drift [deg]")
    axes[1].set_title(f"{LABEL_HEADBAND} rotational drift "
                      f"in {LABEL_G_SHORT.lower()} frame")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "Fig10_Diadem_Drift_Series.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    # Fig11 : par mouvement
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, c, ylabel in [
        (axes[0], "Trans_norm_max_drift_mm", "Max translation drift [mm]"),
        (axes[1], "Rot_drift_max_deg", "Max rotation drift [deg]"),
        (axes[2], "Rot_dev_rms_deg",
         "Rotation deviation about mean offset, RMS [deg]"),
    ]:
        sns.boxplot(data=df_stability, x="Movement", y=c, ax=ax,
                    color="lightsteelblue")
        sns.stripplot(data=df_stability, x="Movement", y=c,
                      color="black", alpha=0.5, size=4, ax=ax)
        ax.set_ylabel(ylabel)
    for tol in TRANSLATION_TOLERANCES_MM:
        axes[0].axhline(tol, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(ROTATION_TOLERANCE_DEG, color="red",
                    linestyle="--", linewidth=1)
    fig.suptitle(f"{LABEL_HEADBAND} drift by movement type")
    plt.tight_layout()
    plt.savefig(OUT / "Fig11_Stability_by_Movement.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    # Fig12 : par sujet
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, c, ylabel in [
        (axes[0], "Trans_norm_max_drift_mm", "Max translation drift [mm]"),
        (axes[1], "Rot_drift_max_deg", "Max rotation drift [deg]"),
        (axes[2], "Rot_dev_rms_deg",
         "Rotation deviation about mean offset, RMS [deg]"),
    ]:
        sns.boxplot(data=df_stability, x="Subject", y=c, ax=ax,
                    color="lightsteelblue")
        sns.stripplot(data=df_stability, x="Subject", y=c,
                      color="black", alpha=0.5, size=4, ax=ax)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
    for tol in TRANSLATION_TOLERANCES_MM:
        axes[0].axhline(tol, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(ROTATION_TOLERANCE_DEG, color="red",
                    linestyle="--", linewidth=1)
    fig.suptitle(f"Inter-subject variability of {LABEL_HEADBAND.lower()} "
                 f"stability")
    plt.tight_layout()
    plt.savefig(OUT / "Fig12_Stability_by_Subject.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_rigidity(df_rigidity):
    if df_rigidity is None or df_rigidity.empty:
        return
    if "Device" not in df_rigidity.columns:
        return

    df_rigidity = df_rigidity.copy()
    df_rigidity["Device"] = df_rigidity["Device"].map(DEVICE_LABEL)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    items_top = [
        ("d12_mean_mm", "M1-M2 mean distance [mm]"),
        ("d13_mean_mm", "M1-M3 mean distance [mm]"),
        ("d23_mean_mm", "M2-M3 mean distance [mm]"),
    ]
    items_bottom = [
        ("Kabsch_rmsd_mean_mm", "Kabsch RMSD on triangle [mm]"),
        ("Height_mm", "Triangle height [mm]"),
        ("Area_mm2", "Triangle area [mm²]"),
    ]

    order = [LABEL_GUTTER, LABEL_HEADBAND]

    for ax, (c, lab) in zip(axes[0], items_top):
        if c not in df_rigidity.columns:
            ax.set_visible(False)
            continue
        d = df_rigidity[["Device", c]].dropna()
        if d.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(data=d, x="Device", y=c, ax=ax, order=order,
                    color="lightsteelblue")
        sns.stripplot(data=d, x="Device", y=c, color="black",
                      alpha=0.35, size=3, ax=ax, order=order)
        ax.set_ylabel(lab)
        ax.set_xlabel("")

    for ax, (c, lab) in zip(axes[1], items_bottom):
        if c not in df_rigidity.columns:
            ax.set_visible(False)
            continue
        d = df_rigidity[["Device", c]].dropna()
        if d.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(data=d, x="Device", y=c, ax=ax, order=order,
                    color="lightsteelblue")
        sns.stripplot(data=d, x="Device", y=c, color="black",
                      alpha=0.35, size=3, ax=ax, order=order)
        ax.set_ylabel(lab)
        ax.set_xlabel("")

    fig.suptitle(f"Rigidity diagnostics: {LABEL_G_SHORT.lower()} triangle "
                 f"vs {LABEL_HEADBAND.lower()} triangle")
    plt.tight_layout()
    plt.savefig(OUT / "Fig14_Rigidity_Diagnostics.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_skin_drift(df_skin):
    if df_skin is None or df_skin.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Qualité du repère cutané
    sns.histplot(df_skin["Skin_RMSD_mean_mm"].dropna(), bins=30,
                 kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].axvline(FIXED_THRESHOLD_MM, color="red", linestyle="--",
                       linewidth=1,
                       label=f"Threshold {FIXED_THRESHOLD_MM} mm")
    axes[0, 0].set_xlabel("Skin reference RMSD [mm]")
    axes[0, 0].set_title("Quality of the skin reference (fit RMSD)")
    axes[0, 0].legend()

    # Nombre de marqueurs fixes
    sns.histplot(df_skin["N_fixed_markers"].dropna(), bins=20,
                 kde=False, ax=axes[0, 1], color="darkorange")
    axes[0, 1].axvline(MIN_FIXED_MARKERS, color="red", linestyle="--",
                       linewidth=1,
                       label=f"MIN_FIXED_MARKERS = {MIN_FIXED_MARKERS}")
    axes[0, 1].set_xlabel("Number of fixed markers used")
    axes[0, 1].set_title("Fixed markers per session")
    axes[0, 1].legend()

    # Translation
    long_t = pandas.DataFrame({
        "Device": ([LABEL_HEADBAND] * len(df_skin)
                   + [LABEL_GUTTER] * len(df_skin)),
        "Trans_max_mm": np.r_[df_skin["D_Trans_norm_max_mm"].values,
                              df_skin["G_Trans_norm_max_mm"].values],
    })
    sns.boxplot(data=long_t, x="Device", y="Trans_max_mm", ax=axes[1, 0],
                order=[LABEL_HEADBAND, LABEL_GUTTER],
                color="lightsteelblue")
    sns.stripplot(data=long_t, x="Device", y="Trans_max_mm",
                  color="black", alpha=0.3, size=3, ax=axes[1, 0],
                  order=[LABEL_HEADBAND, LABEL_GUTTER])
    axes[1, 0].set_ylabel("Max translation drift in skin frame [mm]")
    axes[1, 0].set_title("Translation drift in skin frame")

    # Rotation
    long_r = pandas.DataFrame({
        "Device": ([LABEL_HEADBAND] * len(df_skin)
                   + [LABEL_GUTTER] * len(df_skin)),
        "Rot_max_deg": np.r_[df_skin["D_Rot_max_deg"].values,
                              df_skin["G_Rot_max_deg"].values],
    })
    sns.boxplot(data=long_r, x="Device", y="Rot_max_deg", ax=axes[1, 1],
                order=[LABEL_HEADBAND, LABEL_GUTTER],
                color="lightsteelblue")
    sns.stripplot(data=long_r, x="Device", y="Rot_max_deg",
                  color="black", alpha=0.3, size=3, ax=axes[1, 1],
                  order=[LABEL_HEADBAND, LABEL_GUTTER])
    axes[1, 1].set_ylabel("Max rotation drift in skin frame [deg]")
    axes[1, 1].set_title("Rotation drift in skin frame")

    fig.suptitle("Cross-validation against multi-marker skin reference")
    plt.tight_layout()
    plt.savefig(OUT / "Fig16_Skin_Drift_Comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close()


# =====================================================================
# TABLEAUX
# =====================================================================

REL_METRICS = [
    ("Vector_RMSE", lambda p: col(p, "RMSE"), False),
    ("Norm_RMSE", lambda p: col(p, "RMSE_norm"), False),
    ("NRMSE_pct", lambda p: f"{p}_NRMSE_pct", True),
    ("NRMSE_norm_pct", lambda p: f"{p}_NRMSE_norm_pct", True),
    ("Peak_relerr_pct", lambda p: f"{p}_Peak_relerr_pct", True),
    ("Abs_peak_relerr_pct", lambda p: f"{p}_Peak_relerr_pct", True),
    ("Pearson_norm", lambda p: f"{p}_Pearson_norm", True),
    ("Angle_aligned_deg", lambda p: f"{p}_Angle_mean_deg", True),
    ("Angle_unaligned_deg", lambda p: f"{p}_Angle_mean_unaligned_deg", True),
    ("Vector_RMSE_unaligned", lambda p: col(p, "RMSE_unaligned"), False),
]


def export_tables(df_all, df_main, df_static, df_stability, df_sweep,
                  df_skipped, df_rigidity, df_per_file, df_static_per_file,
                  df_skin, df_skin_fixed):

    df_all.to_csv(OUT / "01_Marqueurs_Comparaison.csv",
                  index=False, encoding="utf-8-sig")

    if not df_stability.empty:
        df_stability.to_csv(OUT / "02_Diademe_Stabilite.csv",
                            index=False, encoding="utf-8-sig")

    if df_rigidity is not None and not df_rigidity.empty:
        df_rigidity.to_csv(OUT / "14_Rigidite_par_Fichier.csv",
                           index=False, encoding="utf-8-sig")
        summary = df_rigidity.groupby("Device").agg(
            N_files=("File", "count"),
            d12_mean=("d12_mean_mm", "median"),
            d12_std_median=("d12_std_mm", "median"),
            d13_mean=("d13_mean_mm", "median"),
            d13_std_median=("d13_std_mm", "median"),
            d23_mean=("d23_mean_mm", "median"),
            d23_std_median=("d23_std_mm", "median"),
            Side_a=("Side_a_mm", "median"),
            Side_b=("Side_b_mm", "median"),
            Side_c=("Side_c_mm", "median"),
            Height=("Height_mm", "median"),
            Area=("Area_mm2", "median"),
            Kabsch_rmsd=("Kabsch_rmsd_mean_mm", "median"),
            Kabsch_rmsd_max=("Kabsch_rmsd_max_mm", "median"),
        ).round(3)
        summary.to_csv(OUT / "15_Rigidite_Resume.csv", encoding="utf-8-sig")

    if not df_per_file.empty:
        df_per_file.to_csv(OUT / "16_Par_Fichier_Moving.csv",
                           index=False, encoding="utf-8-sig")
    if not df_static_per_file.empty:
        df_static_per_file.to_csv(OUT / "17_Par_Fichier_Static.csv",
                                  index=False, encoding="utf-8-sig")

    if df_skin_fixed is not None and not df_skin_fixed.empty:
        df_skin_fixed.to_csv(OUT / "18_Skin_Fixed_Selection.csv",
                             index=False, encoding="utf-8-sig")
    if df_skin is not None and not df_skin.empty:
        df_skin.to_csv(OUT / "19_Skin_Triple_Comparison.csv",
                       index=False, encoding="utf-8-sig")
        df_skin.groupby("Movement").agg(
            N_sessions=("File", "count"),
            N_fixed_median=("N_fixed_markers", "median"),
            Skin_RMSD_median_mm=("Skin_RMSD_mean_mm", "median"),
            Skin_RMSD_max_median_mm=("Skin_RMSD_max_mm", "median"),
            D_Trans_max_median_mm=("D_Trans_norm_max_mm", "median"),
            G_Trans_max_median_mm=("G_Trans_norm_max_mm", "median"),
            D_Rot_max_median_deg=("D_Rot_max_deg", "median"),
            G_Rot_max_median_deg=("G_Rot_max_deg", "median"),
            Trans_ratio_D_over_G=("Trans_ratio_D_over_G", "median"),
            Rot_ratio_D_over_G=("Rot_ratio_D_over_G", "median"),
        ).round(3).to_csv(OUT / "20_Skin_Summary_by_Movement.csv",
                          encoding="utf-8-sig")

    if not df_main.empty:
        agg = {"N_measurements": ("Marker", "count")}
        for p in KIN:
            agg[f"{p}_NRMSE_median_pct"] = (f"{p}_NRMSE_pct", "median")
            agg[f"{p}_RMSE_median"] = (col(p, "RMSE"), "median")
            agg[f"{p}_RMSE_norm_median"] = (col(p, "RMSE_norm"), "median")
            agg[f"{p}_Peak_relerr_median_pct"] = (
                f"{p}_Peak_relerr_pct", "median")
            agg[f"{p}_Angle_mean_deg"] = (f"{p}_Angle_mean_deg", "mean")
            agg[f"{p}_Pearson_norm_median"] = (f"{p}_Pearson_norm", "median")
        df_main.groupby("Movement").agg(**agg).round(4).to_csv(
            OUT / "03_Synthese_par_Mouvement.csv", encoding="utf-8-sig")

        agg = {"N_measurements": ("Movement", "count")}
        for p in KIN:
            agg[f"{p}_NRMSE_mean_pct"] = (f"{p}_NRMSE_pct", "mean")
            agg[f"{p}_RMSE_mean"] = (col(p, "RMSE"), "mean")
            agg[f"{p}_RMSE_norm_mean"] = (col(p, "RMSE_norm"), "mean")
            agg[f"{p}_Bias_X_mean"] = (col(p, "Bias_X"), "mean")
            agg[f"{p}_Bias_Y_mean"] = (col(p, "Bias_Y"), "mean")
            agg[f"{p}_Bias_Z_mean"] = (col(p, "Bias_Z"), "mean")
            agg[f"{p}_Angle_mean_deg"] = (f"{p}_Angle_mean_deg", "mean")
        df_main.groupby("Marker").agg(**agg).round(4).to_csv(
            OUT / "04_Synthese_par_Marqueur.csv", encoding="utf-8-sig")

    groups = {
        "moving_all": df_all[df_all["Marker_class"] == "moving"],
        "moving_QC_ok": df_all[(df_all["Marker_class"] == "moving")
                               & (df_all["QC"] == "ok")],
        "static": df_all[df_all["Marker_class"] == "static"],
    }
    rows = []
    for gname, d in groups.items():
        if d.empty:
            continue
        for p in KIN:
            r = {"Group": gname, "Quantity": p, "Unit": unit_label(p),
                 "N": len(d)}
            for label, fn, relative in REL_METRICS:
                if gname == "static" and relative:
                    r[label] = "n/a (pas de mouvement)"
                    continue
                s = d[fn(p)]
                if label == "Abs_peak_relerr_pct":
                    s = s.abs()
                r[label] = med_iqr(s)
            rows.append(r)
    pandas.DataFrame(rows).to_csv(
        OUT / "06_Synthese_Globale_Cinematique.csv",
        index=False, encoding="utf-8-sig")

    if not df_stability.empty:
        df_stability.groupby("Movement").agg(
            N_files=("File", "count"),
            Trans_mean_mm=("Trans_norm_mean_mm", "mean"),
            Trans_max_median_mm=("Trans_norm_max_drift_mm", "median"),
            Trans_max_SD_mm=("Trans_norm_max_drift_mm", "std"),
            Trans_speed_RMS_median_mm_s=(
                "Trans_drift_speed_RMS_mm_s", "median"),
            Rot_mean_deg=("Rot_drift_mean_deg", "mean"),
            Rot_max_median_deg=("Rot_drift_max_deg", "median"),
            Rot_max_SD_deg=("Rot_drift_max_deg", "std"),
            Frame_offset_median_deg=("Frame_offset_deg", "median"),
            Rot_dev_rms_median_deg=("Rot_dev_rms_deg", "median"),
        ).round(3).to_csv(OUT / "07_Stabilite_par_Mouvement.csv",
                          encoding="utf-8-sig")

        df_stability.groupby("Subject").agg(
            N_sessions=("File", "count"),
            Trans_max_median_mm=("Trans_norm_max_drift_mm", "median"),
            Trans_max_max_mm=("Trans_norm_max_drift_mm", "max"),
            Rot_max_median_deg=("Rot_drift_max_deg", "median"),
            Rot_max_max_deg=("Rot_drift_max_deg", "max"),
            Frame_offset_median_deg=("Frame_offset_deg", "median"),
            Rot_dev_rms_median_deg=("Rot_dev_rms_deg", "median"),
        ).round(3).to_csv(OUT / "08_Stabilite_par_Sujet.csv",
                          encoding="utf-8-sig")

    if not df_static.empty:
        items = [
            ("Pos_SD", "G_Pos_SD_mm", "D_Pos_SD_mm", "mm"),
            ("Vel_RMS", "G_Vel_RMS_mm_s", "D_Vel_RMS_mm_s", "mm/s"),
            ("Acc_RMS", "G_Acc_RMS_mm_s2", "D_Acc_RMS_mm_s2", "mm/s²"),
        ]
        rows = []
        for label, cg, cd, unit in items:
            x = df_static[[cg, cd]].dropna()
            ratio = (x[cd] / x[cg]).replace([np.inf, -np.inf], np.nan)
            try:
                pv = (wilcoxon(x[cg], x[cd])[1]
                      if len(x) >= 6 and (x[cg] - x[cd]).abs().sum() > 0
                      else np.nan)
            except Exception:
                pv = np.nan
            rows.append({
                "Metric": label,
                "Unit": unit,
                "N": len(x),
                "Gutter_median_IQR": med_iqr(x[cg]),
                "Headband_median_IQR": med_iqr(x[cd]),
                "Ratio_D_over_G_median": round(float(ratio.median()), 3),
                "Wilcoxon_p": pv,
            })
        for p in KIN:
            rows.append({
                "Metric": f"{p}_G_vs_D_error_floor",
                "Unit": unit_label(p),
                "N": len(df_static),
                "Gutter_median_IQR": f"vector RMSE (aligned): "
                                     f"{med_iqr(df_static[col(p, 'RMSE')])}",
                "Headband_median_IQR": f"norm RMSE: "
                                       f"{med_iqr(df_static[col(p, 'RMSE_norm')])}",
                "Ratio_D_over_G_median": np.nan,
                "Wilcoxon_p": np.nan,
            })
        pandas.DataFrame(rows).to_csv(
            OUT / "10_Bruit_Statique.csv", index=False, encoding="utf-8-sig")

    if not df_sweep.empty:
        d = main_subset(df_sweep)
        if not d.empty:
            d = d.copy()
            d["Abs_Peak_relerr_pct"] = d["Peak_relerr_pct"].abs()
            d.groupby(["Quantity", "Cutoff_Hz"]).agg(
                N=("Marker", "count"),
                Pearson_norm_median=("Pearson_norm", "median"),
                Abs_peak_relerr_median_pct=("Abs_Peak_relerr_pct", "median"),
                NRMSE_norm_median_pct=("NRMSE_norm_pct", "median"),
                RMSE_norm_median=("RMSE_norm", "median"),
                Peak_G_median=("Peak_G", "median"),
                Peak_D_median=("Peak_D", "median"),
            ).round(4).to_csv(OUT / "11_Sensibilite_Coupure.csv",
                              encoding="utf-8-sig")
        df_sweep.to_csv(OUT / "11b_Sensibilite_Coupure_brut.csv",
                        index=False, encoding="utf-8-sig")

    cov = df_all.groupby("Marker").agg(
        N_analysed=("File", "count"),
        Valid_raw_pct_mean=("Valid_raw_pct", "mean"),
        Run_pct_mean=("Run_pct", "mean"),
        Interp_pct_mean=("Interp_pct", "mean"),
        N_QC_check=("QC", lambda s: int((s == "check").sum())),
    )
    if not df_skipped.empty:
        sk = df_skipped.groupby("Marker").size().rename("N_excluded")
        cov = cov.join(sk, how="outer")
    else:
        cov["N_excluded"] = 0
    cov["N_excluded"] = cov["N_excluded"].fillna(0).astype(int)
    cov.round(2).sort_values("N_excluded", ascending=False).to_csv(
        OUT / "12_Couverture_Marqueurs.csv", encoding="utf-8-sig")

    if not df_skipped.empty:
        df_skipped.to_csv(OUT / "13_Marqueurs_Exclus.csv",
                          index=False, encoding="utf-8-sig")


# =====================================================================
# PROGRAMME PRINCIPAL
# =====================================================================

def main():

    OUT.mkdir(parents=True, exist_ok=True)

    series_store = {}
    stab_series = {}
    all_markers, all_stability, all_sweep, all_skipped = [], [], [], []
    all_rigidity = []
    all_skin = []
    all_skin_fixed = []

    gutter_files = sorted(DIR_G.glob("*.csv"))

    print("\n" + "=" * 80)
    print("ANALYSE CINÉMATIQUE DENTAL SPLINT vs HEADBAND — V9")
    print("=" * 80)
    print(f"Fichiers splint trouvés : {len(gutter_files)}")
    print(f"Dossier de sortie : {OUT}")
    print(f"Repère : {REPERE_MODE} | Baseline : {BASELINE_FRAMES} frames")
    print(f"Filtre : Butterworth {FILTER_CUTOFF_HZ} Hz, ordre {FILTER_ORDER}")
    print(f"Balayage de coupure : {SWEEP_CUTOFFS_HZ} Hz")
    print(f"Seuil marqueurs fixes : {FIXED_THRESHOLD_MM} mm")
    print(f"Nb minimum marqueurs fixes : {MIN_FIXED_MARKERS}")

    for index, gutter_file in enumerate(gutter_files, start=1):

        print(f"\n[{index}/{len(gutter_files)}] {gutter_file.name}")

        diadem_file = find_matching_diadem_file(gutter_file)
        if diadem_file is None:
            print("   ⚠ Aucun fichier diadème correspondant")
            continue

        gutter_data, fs_g = load_vicon(gutter_file)
        diadem_data, fs_d = load_vicon(diadem_file)

        if gutter_data is None or diadem_data is None:
            print("   ⚠ Erreur de lecture")
            continue

        if abs(fs_g - fs_d) > 1e-6:
            print(f"   ⚠ Fréquences différentes (G={fs_g}, D={fs_d}) : ignoré")
            continue

        missing = [m for m in REF_POINTS
                   if m not in gutter_data or m not in diadem_data]
        if missing:
            print(f"   ⚠ Repère incomplet : {missing}")
            continue

        (mres, sres, wres, skp, rres, kres, kfixed) = analyze_file_pair(
            gutter_data, diadem_data, gutter_file.name, fs_g,
            series_store, stab_series)

        all_markers.extend(mres)
        all_stability.extend(sres)
        all_sweep.extend(wres)
        all_skipped.extend(skp)
        all_rigidity.extend(rres)
        all_skin.extend(kres)
        all_skin_fixed.extend(kfixed)

        n_fixed = 0
        if kfixed and kfixed[0]["Fixed_markers"]:
            n_fixed = len(kfixed[0]["Fixed_markers"].split("|"))
        print(f"   ✓ {len(mres)} marqueurs analysés, {len(skp)} exclus, "
              f"{n_fixed} marqueurs fixes pour le repère cutané")

    df_all = pandas.DataFrame(all_markers)
    df_stability = pandas.DataFrame(all_stability)
    df_sweep = pandas.DataFrame(all_sweep)
    df_skipped = pandas.DataFrame(all_skipped)
    df_rigidity = pandas.DataFrame(all_rigidity)
    df_skin = pandas.DataFrame(all_skin)
    df_skin_fixed = pandas.DataFrame(all_skin_fixed)

    print("\n" + "=" * 80)
    print(f"Résultats marqueurs : {len(df_all)}")
    print(f"Résultats stabilité : {len(df_stability)}")
    print(f"Résultats rigidité  : {len(df_rigidity)}")
    print(f"Résultats cutanés   : {len(df_skin)}")

    if df_all.empty:
        print("Aucun résultat marqueur : arrêt.")
        return

    df_main = main_subset(df_all)
    df_static = df_all[df_all["Marker_class"] == "static"]
    df_sweep_main = main_subset(df_sweep) if not df_sweep.empty else df_sweep

    test_cols = []
    for p in KIN:
        test_cols += [col(p, "Peak_G"), col(p, "Peak_D"),
                      f"{p}_NRMSE_pct"]
    df_per_file = aggregate_per_file(df_main, test_cols)

    static_cols = ["G_Pos_SD_mm", "D_Pos_SD_mm",
                   "G_Vel_RMS_mm_s", "D_Vel_RMS_mm_s",
                   "G_Acc_RMS_mm_s2", "D_Acc_RMS_mm_s2"]
    df_static_per_file = (aggregate_per_file(df_static, static_cols)
                          if not df_static.empty else pandas.DataFrame())

    print(f"   dont 'moving' : {len(df_main)} | 'static' : {len(df_static)}")
    print(f"   Essais (fichiers) pour les tests : {len(df_per_file)}")

    export_tables(df_all, df_main, df_static, df_stability, df_sweep,
                  df_skipped, df_rigidity, df_per_file, df_static_per_file,
                  df_skin, df_skin_fixed)

    # =================================================================
    # GÉNÉRATION DES FIGURES
    # =================================================================
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES FIGURES")
    print("=" * 80)

    if not df_main.empty:
        print("   → Fig_BlandAltman_ROM.png (Figure 7 de l'article)")
        fig_bland_altman_rom(df_main)
        print("   → Fig01b_BlandAltman_Peaks.png (supplementary)")
        fig_bland_altman_peaks(df_main)
        print("   → Fig02_NRMSE_by_Movement.png")
        fig_nrmse_by_movement(df_main)
        print("   → Fig06_Example_Norms.png")
        fig_example(df_main, series_store)
    if not df_static.empty:
        print("   → Fig08_Static_Noise_G_vs_D.png")
        fig_static_noise(df_static)
    if not df_sweep_main.empty:
        print("   → Fig13_Peaks_vs_Cutoff.png (Figure 4 de l'article)")
        fig_cutoff_sweep(df_sweep_main)
    if not df_stability.empty:
        print("   → Fig10_Diadem_Drift_Series.png")
        print("   → Fig11_Stability_by_Movement.png")
        print("   → Fig12_Stability_by_Subject.png")
        fig_stability(df_stability, stab_series)
    if not df_rigidity.empty:
        print("   → Fig14_Rigidity_Diagnostics.png (Figure 2 de l'article)")
        fig_rigidity(df_rigidity)
    if not df_skin.empty:
        print("   → Fig16_Skin_Drift_Comparison.png (Figure 11 de l'article)")
        fig_skin_drift(df_skin)

    print("\n" + "=" * 80)
    print("RÉSUMÉ — marqueurs 'moving' (médianes)")
    print("=" * 80)
    for p in KIN:
        if df_main.empty:
            break
        print(f"\n{p} [{unit_label(p)}]")
        print(f"   RMSE vectoriel (aligné)  : "
              f"{df_main[col(p, 'RMSE')].median():.3f}")
        print(f"   NRMSE (% du pic G)       : "
              f"{df_main[f'{p}_NRMSE_pct'].median():.1f}")
        print(f"   Pearson (normes)         : "
              f"{df_main[f'{p}_Pearson_norm'].median():.3f}")

    if not df_skin.empty:
        print(f"\nCross-validation cutanée (n={len(df_skin)} sessions) :")
        print(f"   Skin RMSD median : "
              f"{df_skin['Skin_RMSD_mean_mm'].median():.3f} mm")
        print(f"   Diadème translation max : "
              f"{df_skin['D_Trans_norm_max_mm'].median():.2f} mm")
        print(f"   Splint  translation max : "
              f"{df_skin['G_Trans_norm_max_mm'].median():.2f} mm")
        print(f"   Ratio D/G translation   : "
              f"{df_skin['Trans_ratio_D_over_G'].median():.2f}")
        print(f"   Ratio D/G rotation      : "
              f"{df_skin['Rot_ratio_D_over_G'].median():.2f}")

    print(f"\n✓ Analyse V9 terminée. Résultats dans :\n  {OUT}\n")


if __name__ == "__main__":
    main()