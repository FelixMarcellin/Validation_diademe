# -*- coding: utf-8 -*-
"""
======================================================================
VALIDATION D'INTERCHANGEABILITÉ — DENTAL SPLINT vs HEADBAND — V12.4.1
======================================================================

Objectif
--------
Démontrer que le serre-tête (Headband) peut REMPLACER la gouttière
maxillaire (Dental splint) comme repère de référence pour l'analyse
cinématique faciale.

Démarche
--------
Tous les marqueurs faciaux communs aux deux acquisitions sont inclus
comme candidats dans l'analyse principale. L'inclusion dépend uniquement
de la qualité et de la durée des trajectoires, jamais de l'amplitude du
mouvement. Une classification secondaire décrit chaque marqueur comme
moving, static, discordant (classification différente selon le repère)
ou unclassifiable (amplitude indisponible). Cette classification ne
sert pas à exclure un marqueur de l'analyse principale.

Chaîne de validation (ordre narratif des figures)
-------------------------------------------------
[1] Rigidité du repère      → Fig01_Rigidity_Diagnostics
[2] Glissement sur la tête  → Fig02_Skin_Drift_Comparison
[3] Faible amplitude       → Fig03_FaibleAmplitude_Variabilite
[4] Accord de mouvement     → Fig04_BlandAltman_ROM (+ TOST)
                              Fig05_NRMSE_by_Movement
[5] Robustesse au filtre    → Fig06_Peaks_vs_Cutoff
Analyse anatomique         → 21..26 (zones, côtés, mouvements, participants)
Annexes                     → Fig07..Fig13

Métrologie des marqueurs
------------------------
- Gouttière  : M1, M2, M3 = demi-sphères rétro-réfléchissantes Ø 5 mm
- Serre-tête : M1, M2, M3 = sphères rétro-réfléchissantes Ø 14 mm
              (noms Vicon paramétrables via REF_POINTS_D)
- Face       : sphères rétro-réfléchissantes Ø 1.5 mm

La variabilité des 30 premières images est un proxy exploratoire : elle
peut inclure du mouvement réel et du mouvement de tête, et ne constitue
pas une mesure isolée du bruit instrumental. La classification reste secondaire.

Marge d'équivalence clinique
----------------------------
EQUIVALENCE_DELTA_MM = 2.0 mm sur la ROM 3D. Marge exploratoire à
justifier cliniquement avant publication. Le TOST porte sur les moyennes
par participant (n=9) et teste le biais moyen, pas l'interchangeabilité
individuelle. Les limites d'accord sont estimées par modèle mixte.

======================================================================
"""

from pathlib import Path

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr, wilcoxon, ttest_1samp, spearmanr


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
    r"\RESULTATS\RESULTATS_COMPARAISON_V12_3_VALIDATION_ANATOMIQUE"
)

# --- Cartographie anatomique des 105 marqueurs --------------------------
# Placer zone_marqueurs(6).csv dans le même dossier que ce script.
SCRIPT_DIR = Path(__file__).resolve().parent
ZONE_FILE_CANDIDATES = [
    SCRIPT_DIR / "zone_marqueurs(6).csv",
    SCRIPT_DIR / "zone_marqueurs.csv",
    SCRIPT_DIR / "zone_marqueurs(5).csv",
]
ZONE_FILE = next((p for p in ZONE_FILE_CANDIDATES if p.exists()),
                 ZONE_FILE_CANDIDATES[0])

# --- Repères : noms Vicon des 3 marqueurs de chaque repère -------------
REF_POINTS_G = ["M1", "M2", "M3"]
REF_POINTS_D = ["M1", "M2", "M3"]

# --- Description des mouvements (pour labels de figures / CSV) --------
MOVEMENT_DESCRIPTION = {
    "M1": "Simple eye closure",
    "M2": "Forced eye closure",
    "M3": "Lip protrusion ('o')",
    "M4": "Lip protrusion ('pou')",
    "M5": "Broad smile",
}

# --- Métrologie des marqueurs ------------------------------------------
MARKER_DIAMETER_MM = {
    "gouttiere": {"M1": 5.0, "M2": 5.0, "M3": 5.0},
    "diademe":   {"M1": 14.0, "M2": 14.0, "M3": 14.0},
}
FACE_DIAMETER_MM = 1.5

# --- Classification descriptive moving/static -------------------------
# Cette classification ne détermine plus l'inclusion dans l'analyse.
# Elle sert uniquement aux analyses secondaires / descriptives.
AMP_MOVING_FACTOR = 5.0
AMP_MOVING_FLOOR_MM = 1.0

# --- Sélection des marqueurs fixes pour le repère cutané --------------
# Symétrique : un marqueur est "fixe" si son amplitude reste SOUS le
# seuil FIXED_THRESHOLD_MM vu depuis G ET depuis D.
FIXED_THRESHOLD_MM = 1.5
MIN_FIXED_MARKERS = 10
MIN_SKIN_FRAMES = 30

# --- Marge d'équivalence clinique (TOST) --------------------------------
EQUIVALENCE_DELTA_MM = 2.0   # ROM 3D ; à justifier cliniquement

# --- Acquisition -------------------------------------------------------
FPS_DEFAULT = 100.0

# --- Filtrage / dérivation ---------------------------------------------
FILTER_CUTOFF_HZ = 6.0
FILTER_ORDER = 2
EDGE_TRIM_FRAMES = 5

SWEEP_CUTOFFS_HZ = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# --- Trous et qualité --------------------------------------------------
MAX_GAP_FRAMES = 10
MIN_RUN_PCT = 70.0
MAX_INTERP_PCT = 15.0
USE_QC_FOR_MAIN = False

# --- Validité ----------------------------------------------------------
MIN_VALID_FRAMES = 30
EPS = 1e-9

# --- Ligne de base moyennée --------------------------------------------
BASELINE_FRAMES = 5

# --- Mode de repère ----------------------------------------------------
REPERE_MODE = "kabsch"

# --- Alignement des orientations ---------------------------------------
ALIGN_ORIENTATION = True

MIN_NORM = {"Pos": 1.0, "Vel": 15.0, "Acc": 400.0}

# --- Tolérances --------------------------------------------------------
TRANSLATION_TOLERANCES_MM = [2.0, 3.0]
ROTATION_TOLERANCE_DEG = 2.0

KIN = ["Pos", "Vel", "Acc"]
UNITS = {"Pos": "mm", "Vel": "mm_s", "Acc": "mm_s2"}
LABELS = {
    "Pos": "Displacement [mm]",
    "Vel": "Speed [mm/s]",
    "Acc": "Acceleration [mm/s²]",
}

# --- Labels pour publication -------------------------------------------
LABEL_GUTTER = "Dental splint"
LABEL_HEADBAND = "Headband"
LABEL_G_SHORT = "Splint"
LABEL_D_SHORT = "Headband"

DEVICE_LABEL = {
    "Gutter": LABEL_GUTTER,
    "Headband": LABEL_HEADBAND,
}

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


# =====================================================================
# MÉTROLOGIE — PLANCHER DE BRUIT DES MARQUEURS
# =====================================================================

def early_window_global_sd(data, marker, n_frames=30):
    """SD brute globale des 30 premières images : proxy, pas bruit instrumental isolé."""
    if marker not in data:
        return np.nan
    xyz = data[marker][:n_frames]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 5:
        return np.nan
    return float(np.mean(np.std(xyz, axis=0)))


def measure_early_window_variability(gutter_data, diadem_data, facial_markers):
    rows = []
    for m in REF_POINTS_G:
        if m in gutter_data:
            rows.append({
                "Role": "Reference", "Device": "Gutter", "Marker": m,
                "Diameter_mm": MARKER_DIAMETER_MM["gouttiere"].get(m, np.nan),
                "EarlyWindow_SD_mm": early_window_global_sd(gutter_data, m),
            })
    for m in REF_POINTS_D:
        if m in diadem_data:
            rows.append({
                "Role": "Reference", "Device": "Headband", "Marker": m,
                "Diameter_mm": MARKER_DIAMETER_MM["diademe"].get(m, np.nan),
                "EarlyWindow_SD_mm": early_window_global_sd(diadem_data, m),
            })
    for m in facial_markers:
        if m in gutter_data:
            rows.append({
                "Role": "Facial", "Device": "Gutter", "Marker": m,
                "Diameter_mm": FACE_DIAMETER_MM,
                "EarlyWindow_SD_mm": early_window_global_sd(gutter_data, m),
            })
        if m in diadem_data:
            rows.append({
                "Role": "Facial", "Device": "Headband", "Marker": m,
                "Diameter_mm": FACE_DIAMETER_MM,
                "EarlyWindow_SD_mm": early_window_global_sd(diadem_data, m),
            })
    return rows


# =====================================================================
# STATISTIQUES D'ÉQUIVALENCE
# =====================================================================

def equivalence_test_tost_subject_means(subject_differences, delta):
    """TOST sur les moyennes par participant (unité indépendante = sujet).

    Ce test porte uniquement sur le biais moyen entre méthodes. Il ne démontre
    pas l'interchangeabilité individuelle, qui nécessite aussi des LoA acceptables.
    Retourne la p-value TOST et l'IC à 90 % de la moyenne des différences.
    """
    values = np.asarray(subject_differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return np.nan, False, np.nan, np.nan, n
    lo = ttest_1samp(values, -delta, alternative="greater")
    hi = ttest_1samp(values, +delta, alternative="less")
    p_tost = max(float(lo.pvalue), float(hi.pvalue))
    se = np.std(values, ddof=1) / np.sqrt(n)
    from scipy.stats import t as t_dist
    tcrit = t_dist.ppf(0.95, df=n - 1)  # IC bilatéral 90 %, cohérent avec TOST α=.05
    m = float(np.mean(values))
    return p_tost, bool(p_tost < 0.05), m - tcrit * se, m + tcrit * se, n


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
    """Triangle de référence moyen sur les mêmes images valides pour M1-M3."""
    if not all(name in data for name in ref_points):
        return None
    n = min([n_frames] + [len(data[name]) for name in ref_points])
    if n < 1:
        return None
    common_valid = np.ones(n, dtype=bool)
    for name in ref_points:
        common_valid &= np.isfinite(data[name][:n]).all(axis=1)
    frames = np.flatnonzero(common_valid)[:baseline_frames]
    if len(frames) == 0:
        return None
    return np.array([np.mean(data[name][frames], axis=0)
                     for name in ref_points], dtype=float)


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
    n_use = min(n_frames, len(positions), len(R_frames), len(origin_frames))
    for f in range(n_use):
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
    arr = np.asarray(local, dtype=float)
    # Excursion 3D basée sur la diagonale de la boîte englobante.
    # Contrairement à la médiane des PTP par axe, cette mesure ne devient
    # pas artificiellement nulle lorsqu'un marqueur se déplace surtout
    # selon un seul axe.
    ptp_xyz = np.ptp(arr, axis=0)
    return float(np.linalg.norm(ptp_xyz))


def classify_markers_per_trial(
    gutter_data, diadem_data, facial,
    Rg_list, Og_list, Rd_list, Od_list, n_frames,
    early_sd_face,
):
    """Classification descriptive moving/static pour CET essai.

    La classification ne détermine pas l'inclusion dans l'analyse principale.
    Le statut moving est attribué si l'amplitude dépasse le seuil dans les
    deux repères. Les autres classes sont attribuées par l'appelant.
    """
    threshold = max(
        AMP_MOVING_FACTOR * (early_sd_face if np.isfinite(early_sd_face)
                             else 0.0),
        AMP_MOVING_FLOOR_MM,
    )
    amp_records = []
    moving_G, moving_D = set(), set()
    for m in facial:
        amp_g = amplitude_in_frame(
            gutter_data[m], Rg_list, Og_list, n_frames)
        amp_d = amplitude_in_frame(
            diadem_data[m], Rd_list, Od_list, n_frames)
        amp_records.append({
            "Marker": m,
            "Amp_G_mm": amp_g,
            "Amp_D_mm": amp_d,
            "Threshold_mm": threshold,
        })
        if np.isfinite(amp_g) and amp_g > threshold:
            moving_G.add(m)
        if np.isfinite(amp_d) and amp_d > threshold:
            moving_D.add(m)

    # Classification descriptive uniquement : elle ne conditionne plus
    # l'inclusion dans l'analyse cinématique principale.
    moving = sorted(moving_G & moving_D)
    return moving, amp_records, threshold


def build_skin_reference(face_positions, fixed_markers, n_frames,
                         baseline_frames=BASELINE_FRAMES,
                         min_markers=MIN_FIXED_MARKERS):
    if len(fixed_markers) < min_markers:
        return None
    ref_clouds = []
    used_markers = []
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
        used_markers.append(m)
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
        for i, m in enumerate(used_markers):
            if f >= len(face_positions[m]):
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
        "R_skin": R_skin, "O_skin": O_skin, "rmsd": rmsd,
        "n_markers_used": M, "ref_cloud": ref_cloud,
        "fixed_markers": used_markers,
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

def rigidity_diagnostics(data, fs, file_name, subject, movement, ref_points):
    if not all(m in data for m in ref_points):
        return None
    m1 = data[ref_points[0]]
    m2 = data[ref_points[1]]
    m3 = data[ref_points[2]]
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
        "File": file_name, "Subject": subject, "Movement": movement,
        "N_frames": int(valid.sum()),
        "d12_mean_mm": float(np.mean(d12)),
        "d12_std_mm": float(np.std(d12)),
        "d13_mean_mm": float(np.mean(d13)),
        "d13_std_mm": float(np.std(d13)),
        "d23_mean_mm": float(np.mean(d23)),
        "d23_std_mm": float(np.std(d23)),
        "Side_a_mm": float(a), "Side_b_mm": float(b),
        "Side_c_mm": float(c), "Height_mm": float(h),
        "Area_mm2": float(area),
        "Kabsch_rmsd_mean_mm": float(np.mean(rmsd)),
        "Kabsch_rmsd_std_mm": float(np.std(rmsd)),
        "Kabsch_rmsd_max_mm": float(np.max(rmsd)),
        "Kabsch_note": "3-point rigid fit; residual has limited independent degrees of freedom; interpret alongside inter-marker distances",
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
        out[f"{prefix}_RMSE_{axis_name}_{u}"] = np.sqrt(
            np.mean(err[:, i] ** 2))
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
    trial_classification, amp_details = [], []

    common = [m for m in gutter_data if m in diadem_data]
    facial = [m for m in common
              if m not in REF_POINTS_G and m not in REF_POINTS_D]

    if not common:
        return (marker_results, stability_results, sweep_results, skipped,
                rigidity_results, skin_results, skin_fixed_selection,
                trial_classification, amp_details)

    # La durée de l'essai est déterminée par les six marqueurs de référence.
    # Un marqueur facial plus court ne doit pas tronquer toutes les autres séries.
    required_refs = list(dict.fromkeys(REF_POINTS_G + REF_POINTS_D))
    if not all(m in gutter_data and m in diadem_data for m in required_refs):
        return (marker_results, stability_results, sweep_results, skipped,
                rigidity_results, skin_results, skin_fixed_selection,
                trial_classification, amp_details)
    n_frames = min([len(gutter_data[m]) for m in REF_POINTS_G]
                   + [len(diadem_data[m]) for m in REF_POINTS_D])

    # --- Rigidité --------------------------------------------------
    rig_g = rigidity_diagnostics(
        gutter_data, fs, file_name, subject, movement, REF_POINTS_G)
    rig_d = rigidity_diagnostics(
        diadem_data, fs, file_name, subject, movement, REF_POINTS_D)
    if rig_g is not None:
        rig_g["Device"] = "Gutter"
        rigidity_results.append(rig_g)
    if rig_d is not None:
        rig_d["Device"] = "Headband"
        rigidity_results.append(rig_d)

    # --- Repères frame par frame ------------------------------------
    if REPERE_MODE == "kabsch":
        ref_tri_g = build_reference_triangle(
            gutter_data, REF_POINTS_G, n_frames, BASELINE_FRAMES)
        ref_tri_d = build_reference_triangle(
            diadem_data, REF_POINTS_D, n_frames, BASELINE_FRAMES)
        if ref_tri_g is None or ref_tri_d is None:
            return (marker_results, stability_results, sweep_results,
                    skipped, rigidity_results, skin_results,
                    skin_fixed_selection, trial_classification, amp_details)
        ref_basis_g, _ = get_local_basis_m1m2m3(*ref_tri_g)
        ref_basis_d, _ = get_local_basis_m1m2m3(*ref_tri_d)
        if ref_basis_g is None or ref_basis_d is None:
            return (marker_results, stability_results, sweep_results,
                    skipped, rigidity_results, skin_results,
                    skin_fixed_selection, trial_classification, amp_details)

    g_frames, d_frames = [], []
    for f in range(n_frames):
        if REPERE_MODE == "kabsch":
            Rg, og = get_local_basis_kabsch(
                gutter_data[REF_POINTS_G[0]][f],
                gutter_data[REF_POINTS_G[1]][f],
                gutter_data[REF_POINTS_G[2]][f],
                ref_tri_g, ref_basis_g)
            Rd, od = get_local_basis_kabsch(
                diadem_data[REF_POINTS_D[0]][f],
                diadem_data[REF_POINTS_D[1]][f],
                diadem_data[REF_POINTS_D[2]][f],
                ref_tri_d, ref_basis_d)
        else:
            Rg, og = get_local_basis_m1m2m3(
                gutter_data[REF_POINTS_G[0]][f],
                gutter_data[REF_POINTS_G[1]][f],
                gutter_data[REF_POINTS_G[2]][f])
            Rd, od = get_local_basis_m1m2m3(
                diadem_data[REF_POINTS_D[0]][f],
                diadem_data[REF_POINTS_D[1]][f],
                diadem_data[REF_POINTS_D[2]][f])
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

    # --- Variabilité initiale faciale (proxy exploratoire pour classification) ---
    early_variability_samples = []
    for m in facial:
        s = early_window_global_sd(gutter_data, m)
        if np.isfinite(s):
            early_variability_samples.append(s)
    early_sd_face = (float(np.median(early_variability_samples))
                     if early_variability_samples else np.nan)

    # --- Classification data-driven moving/static ------------------
    moving_markers, amp_records, amp_threshold = classify_markers_per_trial(
        gutter_data, diadem_data, facial,
        Rg_list, Og_list, Rd_list, Od_list, n_frames,
        early_sd_face)

    marker_classes = {}
    for rec in amp_records:
        ag, ad = rec["Amp_G_mm"], rec["Amp_D_mm"]
        if not (np.isfinite(ag) and np.isfinite(ad)):
            cls = "unclassifiable"
        elif ag > amp_threshold and ad > amp_threshold:
            cls = "moving"
        elif ag <= amp_threshold and ad <= amp_threshold:
            cls = "static"
        else:
            cls = "discordant"
        marker_classes[rec["Marker"]] = cls

    moving_set = {m for m, cls in marker_classes.items() if cls == "moving"}
    static_markers = [m for m, cls in marker_classes.items() if cls == "static"]
    discordant_markers = [m for m, cls in marker_classes.items() if cls == "discordant"]
    unclassifiable_markers = [m for m, cls in marker_classes.items()
                              if cls == "unclassifiable"]

    trial_classification.append({
        "File": file_name,
        "Subject": subject,
        "Movement": movement,
        "Movement_description": MOVEMENT_DESCRIPTION.get(movement, ""),
        "EarlyWindow_SD_face_mm": early_sd_face,
        "Amp_threshold_mm": amp_threshold,
        "N_facial": len(facial),
        "N_analyzed": 0,
        "N_excluded_trajectory": 0,
        "N_moving": len(moving_markers),
        "N_static": len(static_markers),
        "N_discordant": len(discordant_markers),
        "N_unclassifiable": len(unclassifiable_markers),
        "Moving_markers": "|".join(moving_markers),
        "Static_markers": "|".join(static_markers),
        "Discordant_markers": "|".join(discordant_markers),
        "Unclassifiable_markers": "|".join(unclassifiable_markers),
    })

    for rec in amp_records:
        amp_details.append({
            "File": file_name,
            "Subject": subject,
            "Movement": movement,
            "Marker": rec["Marker"],
            "Amp_G_mm": rec["Amp_G_mm"],
            "Amp_D_mm": rec["Amp_D_mm"],
            "Threshold_mm": rec["Threshold_mm"],
            "Class": marker_classes[rec["Marker"]],
        })

    # --- Marqueurs fixes pour le repère cutané ---------------------
    fixed_markers = []
    for rec in amp_records:
        ag, ad = rec["Amp_G_mm"], rec["Amp_D_mm"]
        if (np.isfinite(ag) and np.isfinite(ad)
                and ag < FIXED_THRESHOLD_MM and ad < FIXED_THRESHOLD_MM):
            fixed_markers.append(rec["Marker"])

    skin_fixed_selection.append({
        "File": file_name, "Subject": subject, "Movement": movement,
        "N_fixed_markers": len(fixed_markers),
        "Fixed_markers": "|".join(fixed_markers) if fixed_markers else "",
        "Threshold_mm": FIXED_THRESHOLD_MM,
    })

    skin_ref = None
    if len(fixed_markers) >= MIN_FIXED_MARKERS:
        skin_ref = build_skin_reference(
            gutter_data, fixed_markers, n_frames,
            baseline_frames=BASELINE_FRAMES,
            min_markers=MIN_FIXED_MARKERS)

    # --- Pose relative G / D ---------------------------------------
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
                rigidity_results, skin_results, skin_fixed_selection,
                trial_classification, amp_details)

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
        "File": file_name, "Subject": subject, "Movement": movement,
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
        "fs": fs, "Trans_norm": drift_norm,
        "Trans_X": drift[:, 0], "Trans_Y": drift[:, 1],
        "Trans_Z": drift[:, 2], "Rot": rot_drift,
    }

    # --- Dérive croisée vs repère cutané ---------------------------
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
                "File": file_name, "Subject": subject, "Movement": movement,
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

    # --- Marqueurs faciaux -----------------------------------------
    min_len = max(MIN_VALID_FRAMES, 2 * EDGE_TRIM_FRAMES + 10)

    for marker in facial:
        # Tous les marqueurs faciaux sont candidats à l'analyse principale.
        # La classe amplitude est conservée comme variable descriptive.
        mclass = marker_classes.get(marker, "unclassifiable")
        pg = np.full((n_frames, 3), np.nan)
        pd_ = np.full((n_frames, 3), np.nan)
        for f in range(n_frames):
            if (f >= len(gutter_data[marker]) or f >= len(diadem_data[marker])
                    or g_frames[f] is None or d_frames[f] is None):
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
                "File": file_name, "Subject": subject, "Movement": movement,
                "Marker": marker, "Marker_class": mclass,
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
            "File": file_name, "Subject": subject, "Movement": movement,
            "Marker": marker, "Marker_class": mclass,
            # Indicateur réservé aux analyses secondaires des métriques relatives.
            # Il ne conditionne jamais l'inclusion dans l'analyse principale.
            "Relative_metrics_interpretable": bool(mclass == "moving"),
            "QC": qc, "Fs_Hz": fs,
            "N_frames_run": run_len,
            "N_frames_analysed": len(kin_g["Pos"]),
            "Valid_raw_pct": valid_raw_pct,
            "Run_pct": run_pct, "Interp_pct": interp_pct,
            "G_Pos_SD_mm": spr_g["Pos"], "D_Pos_SD_mm": spr_d["Pos"],
            "G_Vel_RMS_mm_s": spr_g["Vel"], "D_Vel_RMS_mm_s": spr_d["Vel"],
            "G_Acc_RMS_mm_s2": spr_g["Acc"], "D_Acc_RMS_mm_s2": spr_d["Acc"],
        }
        # Amplitude dynamique propre à chaque grandeur : maximum de la norme
        # de la trajectoire/vectorielle sur la fenêtre analysée. Pour la position,
        # les trajectoires sont déjà recentrées sur la baseline.
        for p in KIN:
            n_g = np.linalg.norm(kin_g[p], axis=1)
            n_d = np.linalg.norm(D_al[p], axis=1)
            row[f"{p}_Amplitude_G"] = float(np.max(n_g)) if len(n_g) else np.nan
            row[f"{p}_Amplitude_D"] = float(np.max(n_d)) if len(n_d) else np.nan
            row[f"{p}_Amplitude_mean"] = float((np.max(n_g) + np.max(n_d)) / 2.0) if len(n_g) and len(n_d) else np.nan
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
                    "File": file_name, "Subject": subject,
                    "Movement": movement, "Marker": marker,
                    "Marker_class": mclass, "QC": qc,
                    "Cutoff_Hz": fc, "Quantity": p,
                    "RMSE_norm": rmse_n,
                    "Pearson_norm": safe_pearson(nG, nD),
                    "Peak_G": peak_g, "Peak_D": peak_d,
                    "Peak_relerr_pct": (
                        100.0 * (peak_d - peak_g) / peak_g
                        if peak_g > EPS else np.nan),
                    "NRMSE_norm_pct": (
                        100.0 * rmse_n / peak_g if peak_g > EPS else np.nan),
                })

    if trial_classification:
        trial_classification[-1]["N_analyzed"] = len(marker_results)
        trial_classification[-1]["N_excluded_trajectory"] = len(skipped)

    return (marker_results, stability_results, sweep_results, skipped,
            rigidity_results, skin_results, skin_fixed_selection,
            trial_classification, amp_details)


# =====================================================================
# SOUS-ENSEMBLES
# =====================================================================

def main_subset(df):
    """Analyse principale : tous les marqueurs avec trajectoire exploitable.

    La classe moving/static/discordant/unclassifiable ne sert pas de critère
    d'inclusion. Le filtrage QC reste optionnel et indépendant de l'amplitude.
    """
    if df.empty:
        return df
    d = df.copy()
    if USE_QC_FOR_MAIN and "QC" in d.columns:
        d = d[d["QC"] == "ok"]
    return d


def med_iqr(s):
    s = pandas.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return "n/a"
    return f"{s.median():.2f} [{s.quantile(.25):.2f}–{s.quantile(.75):.2f}]"


def aggregate_per_file(df, value_cols,
                       group_cols=("File", "Subject", "Movement")):
    if df.empty:
        return df
    avail = [c for c in value_cols if c in df.columns]
    if not avail:
        return pandas.DataFrame()
    return df.groupby(list(group_cols))[avail].median().reset_index()


# =====================================================================
# BLAND–ALTMAN (MODÈLE MIXTE + TOST AU NIVEAU PARTICIPANT)
# =====================================================================

def bland_altman(df, col_g, col_d):
    """Bland–Altman à mesures répétées : sujet + essai + identité marqueur.

    Le biais est l'intercept fixe. L'écart-type total combine les composantes
    sujet, essai, marqueur et résiduelle. Aucun repli vers un test indépendant
    n'est utilisé si le modèle mixte échoue.
    """
    d = df[["Subject", "File", "Marker", col_g, col_d]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10 or d["Subject"].nunique() < 3:
        return None, None
    d["Mean"] = (d[col_g] + d[col_d]) / 2.0
    d["Diff"] = d[col_g] - d[col_d]  # convention : gouttière − diadème
    result = {
        "Method": "mixed-effects_subject_trial_marker",
        "N_obs": int(len(d)),
        "N_subjects": int(d["Subject"].nunique()),
        "N_trials": int(d["File"].nunique()),
        "N_markers": int(d["Marker"].nunique()),
        "Difference_convention": "Gutter_minus_Headband",
        "Model_converged": False,
        "Bias": np.nan, "Bias_CI95_lower": np.nan,
        "Bias_CI95_upper": np.nan, "SD_subject": np.nan,
        "SD_trial": np.nan, "SD_marker": np.nan,
        "SD_residual": np.nan, "SD_total": np.nan,
        "LoA_lower": np.nan, "LoA_upper": np.nan,
        "Model_error": "",
    }
    try:
        import statsmodels.formula.api as smf
        # Variance components croisés : sujet, essai (identifiant de fichier
        # unique donc imbriqué dans le sujet) et identité du marqueur.
        # Un groupe constant est requis par statsmodels pour les effets croisés.
        model = smf.mixedlm(
            "Diff ~ 1", data=d, groups=np.ones(len(d)), re_formula="0",
            vc_formula={"Subject": "0 + C(Subject)",
                        "Trial": "0 + C(File)",
                        "Marker": "0 + C(Marker)"})
        fit = model.fit(reml=True, method=["lbfgs", "powell"],
                        maxiter=2000, disp=False)
        if not fit.converged:
            raise RuntimeError("mixed model did not converge")
        ci = fit.conf_int().loc["Intercept"]
        vc = dict(zip(fit.model.exog_vc.names,
                      np.asarray(fit.vcomp, dtype=float)))
        var_subject = max(0.0, float(vc.get("Subject", 0.0)))
        var_trial = max(0.0, float(vc.get("Trial", 0.0)))
        var_marker = max(0.0, float(vc.get("Marker", 0.0)))
        var_resid = max(0.0, float(fit.scale))
        sd_total = float(np.sqrt(var_subject + var_trial + var_marker + var_resid))
        result.update({
            "Model_converged": True,
            "Bias": float(fit.fe_params["Intercept"]),
            "Bias_CI95_lower": float(ci.iloc[0]),
            "Bias_CI95_upper": float(ci.iloc[1]),
            "SD_subject": np.sqrt(var_subject),
            "SD_trial": np.sqrt(var_trial),
            "SD_marker": np.sqrt(var_marker),
            "SD_residual": np.sqrt(var_resid),
            "SD_total": sd_total,
            "LoA_lower": float(fit.fe_params["Intercept"] - 1.96 * sd_total),
            "LoA_upper": float(fit.fe_params["Intercept"] + 1.96 * sd_total),
        })
    except Exception as error:
        result["Method"] = "mixed_model_failed_no_inferential_fallback"
        result["Model_error"] = f"{type(error).__name__}: {error}"
        # Statistiques classiques conservées uniquement comme descriptif de secours.
        bias = float(d["Diff"].mean())
        sd = float(d["Diff"].std(ddof=1))
        result.update({"Bias": bias, "SD_total": sd,
                       "LoA_lower": bias - 1.96 * sd,
                       "LoA_upper": bias + 1.96 * sd})
    return d, result


# =====================================================================
# FIGURES
# =====================================================================

def _movement_axis_labels(ax, movements_in_data):
    order = sorted(movements_in_data)
    labels = [f"{mv}\n{MOVEMENT_DESCRIPTION.get(mv, '')}"
              for mv in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    return order


def fig_rigidity(df_rigidity):
    """[1/5] Rigidité du repère serre-tête."""
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
    fig.suptitle(
        f"[1/5] Rigidity diagnostics: {LABEL_G_SHORT.lower()} triangle "
        f"(Ø5 mm markers) vs {LABEL_HEADBAND.lower()} triangle "
        f"(Ø14 mm markers)")
    plt.tight_layout()
    plt.savefig(OUT / "Fig01_Rigidity_Diagnostics.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_skin_drift(df_skin):
    """[2/5] Glissement du serre-tête sur la tête."""
    if df_skin is None or df_skin.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    sns.histplot(df_skin["Skin_RMSD_mean_mm"].dropna(), bins=30,
                 kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].axvline(FIXED_THRESHOLD_MM, color="red", linestyle="--",
                       linewidth=1,
                       label=f"Threshold {FIXED_THRESHOLD_MM} mm")
    axes[0, 0].set_xlabel("Skin reference RMSD [mm]")
    axes[0, 0].set_title("Quality of the skin reference (fit RMSD)")
    axes[0, 0].legend()
    sns.histplot(df_skin["N_fixed_markers"].dropna(), bins=20,
                 kde=False, ax=axes[0, 1], color="darkorange")
    axes[0, 1].axvline(MIN_FIXED_MARKERS, color="red", linestyle="--",
                       linewidth=1,
                       label=f"MIN_FIXED_MARKERS = {MIN_FIXED_MARKERS}")
    axes[0, 1].set_xlabel("Number of fixed markers (G∩D)")
    axes[0, 1].set_title("Fixed markers per session (intersection)")
    axes[0, 1].legend()
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
    fig.suptitle("[2/5] Cross-validation against multi-marker skin reference")
    plt.tight_layout()
    plt.savefig(OUT / "Fig02_Skin_Drift_Comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_static_noise(df_static):
    """[3/5] Variabilité cinématique du sous-groupe de faible amplitude."""
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
    fig.suptitle("[3/5] Kinematic variability in low-amplitude markers (exploratory)")
    plt.tight_layout()
    plt.savefig(OUT / "Fig03_FaibleAmplitude_Variabilite.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_bland_altman_rom(df_main):
    """Accord ROM 3D sur tous les marqueurs exploitables.

    LoA issues du modèle mixte sujet/essai/marqueur. Le TOST est calculé sur
    les moyennes individuelles des 9 participants et porte sur le biais moyen.
    """
    p = "Pos"
    cg, cd = col(p, "ROM_G"), col(p, "ROM_D")
    if cg not in df_main.columns or cd not in df_main.columns:
        print(f"[WARN] Colonnes {cg} / {cd} absentes — figure ROM ignorée")
        return
    d, stats = bland_altman(df_main, cg, cd)
    if d is None:
        print("[WARN] Données insuffisantes pour Bland–Altman ROM")
        return

    subject_means = d.groupby("Subject")["Diff"].mean()
    p_tost, equivalent, ci90_lo, ci90_hi, n_subjects = \
        equivalence_test_tost_subject_means(subject_means.values,
                                            EQUIVALENCE_DELTA_MM)
    stats.update({
        "Quantity": "ROM_3D",
        "Equivalence_delta_mm": EQUIVALENCE_DELTA_MM,
        "TOST_unit": "participant_mean_difference",
        "TOST_N_subjects": n_subjects,
        "TOST_mean_bias_mm": float(subject_means.mean()),
        "TOST_CI90_bias_lower_mm": ci90_lo,
        "TOST_CI90_bias_upper_mm": ci90_hi,
        "TOST_p": p_tost,
        "Mean_bias_equivalent": equivalent,
        "Interchangeability_demonstrated": False,
        "Interpretation_note": "TOST evaluates mean bias only; interchangeability also requires acceptable LoA and clinical margins.",
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(d["Mean"], d["Diff"], alpha=0.35, s=15,
               color="steelblue", edgecolor="none")
    bias = stats["Bias"]
    loa_up, loa_dn = stats["LoA_upper"], stats["LoA_lower"]
    model_label = "Mixed-model" if stats["Model_converged"] else "Pooled descriptive"
    ax.axhline(bias, color="black", linewidth=2,
               label=f"{model_label} bias = {bias:.2f} mm")
    ax.axhline(loa_up, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Upper {model_label} LoA = {loa_up:.2f} mm")
    ax.axhline(loa_dn, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Lower {model_label} LoA = {loa_dn:.2f} mm")
    ax.axhspan(-EQUIVALENCE_DELTA_MM, EQUIVALENCE_DELTA_MM,
               color="green", alpha=0.08,
               label=f"Exploratory mean-bias margin ±{EQUIVALENCE_DELTA_MM} mm")
    tost_txt = (f"Subject-level TOST (mean bias only): p = {p_tost:.3f} — "
                f"{'within margin' if equivalent else 'not within margin'}")
    ax.text(0.02, 0.98, tost_txt, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_xlabel("Mean ROM 3D [mm]")
    ax.set_ylabel(f"Difference {LABEL_GUTTER} − {LABEL_HEADBAND} [mm]")
    ax.set_title("Bland–Altman: 3D ROM — all analysable facial markers")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "Fig04_BlandAltman_ROM.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    pandas.DataFrame([stats]).to_csv(
        OUT / "Fig04_BlandAltman_ROM_stats.csv", index=False,
        encoding="utf-8-sig")
    print(f"   ✓ Fig04 ROM: mixed bias={bias:.2f} mm, "
          f"LoA=[{loa_dn:.2f}, {loa_up:.2f}], "
          f"TOST sujet p={p_tost:.3f}")


def fig_nrmse_by_movement(df_main):
    """[4b] Erreur normalisée, sous-groupe secondaire moving uniquement."""
    df_main = df_main[df_main["Relative_metrics_interpretable"]].copy()
    if df_main.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    movements = df_main["Movement"].unique()
    order_mv = sorted(movements)
    for ax, p in zip(axes, KIN):
        c = f"{p}_NRMSE_pct"
        sns.boxplot(data=df_main, x="Movement", y=c, ax=ax,
                    order=order_mv, color="lightsteelblue")
        sns.stripplot(data=df_main, x="Movement", y=c, color="black",
                      alpha=0.35, size=3, ax=ax, order=order_mv)
        ax.set_ylabel("NRMSE = vector RMSE / splint peak [%]")
        ax.set_title(f"{p} — normalised error")
        _movement_axis_labels(ax, movements)
    fig.suptitle("[4b/5] Normalised error — secondary analysis "
                 "(markers above the movement threshold in both frames)")
    plt.tight_layout()
    plt.savefig(OUT / "Fig05_NRMSE_by_Movement.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_cutoff_sweep(df_sweep_main):
    """[5/5] Robustesse au filtre."""
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
    fig.suptitle("[5/5] Absolute peaks across cutoffs "
                 "(stabilisation = no real-signal loss)")
    plt.tight_layout()
    plt.savefig(OUT / "Fig06_Peaks_vs_Cutoff.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_noise_and_amplitude(df_noise, df_amp):
    """[annexe] Proxy de variabilité initiale et amplitudes (exploratoire)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Panel gauche : bruit par type de marqueur
    ax = axes[0]
    if df_noise is not None and not df_noise.empty:
        df_noise = df_noise.copy()
        df_noise["Group"] = (df_noise["Role"] + " — " + df_noise["Device"]
                             + " (Ø" + df_noise["Diameter_mm"].astype(str)
                             + " mm)")
        order = sorted(df_noise["Group"].dropna().unique())
        sns.boxplot(data=df_noise, x="Group", y="EarlyWindow_SD_mm", ax=ax,
                    order=order, color="lightsteelblue")
        sns.stripplot(data=df_noise, x="Group", y="EarlyWindow_SD_mm",
                      color="black", alpha=0.3, size=3, ax=ax, order=order)
        ax.axhline(FIXED_THRESHOLD_MM, color="red", linestyle="--",
                   linewidth=1,
                   label=f"FIXED_THRESHOLD_MM = {FIXED_THRESHOLD_MM} mm")
        ax.set_ylabel("Global SD, first 30 frames [mm]")
        ax.set_title("Initial-window variability proxy by marker/device")
        ax.legend()
        ax.tick_params(axis="x", rotation=35)

    # Panel droit : distribution des amplitudes par mouvement
    ax = axes[1]
    if df_amp is not None and not df_amp.empty:
        df_amp = df_amp.copy()
        movements = sorted(df_amp["Movement"].unique())
        thr_by_mv = df_amp.groupby("Movement")["Threshold_mm"].median()
        sns.boxplot(data=df_amp, x="Movement", y="Amp_G_mm", ax=ax,
                    order=movements, color="lightsteelblue")
        for i, mv in enumerate(movements):
            thr = thr_by_mv.get(mv, np.nan)
            if np.isfinite(thr):
                ax.hlines(thr, i - 0.4, i + 0.4, color="red",
                          linestyle="--", linewidth=2,
                          label="Amp threshold" if i == 0 else None)
        ax.set_ylabel("Marker amplitude in local frame [mm]")
        ax.set_title("Early-window variability proxy and movement amplitude (exploratory)")
        ax.legend()
        _movement_axis_labels(ax, movements)

    fig.suptitle("[annexe] Initial-window variability proxy & amplitude distribution")
    plt.tight_layout()
    plt.savefig(OUT / "Fig07_Variabilite_Initiale_Amplitude.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_bland_altman_peaks(df_main):
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
        ba_rows.append({"Quantity": p, "Population": "moving_secondary", **res})
    fig.suptitle("Bland–Altman on peak norm (moving markers)")
    plt.tight_layout()
    plt.savefig(OUT / "Fig08_BlandAltman_Peaks.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    if ba_rows:
        pandas.DataFrame(ba_rows).to_csv(
            OUT / "05_BlandAltman_Peaks.csv", index=False,
            encoding="utf-8-sig")


def fig_example(df_main, series_store):
    for mv, mk in [("M5", "C02"), ("M4", "Pps01"), ("M3", "Pps01"),
                   ("M1", "Lc2"), ("M2", "Lc2")]:
        sel = df_main[(df_main["Movement"] == mv) & (df_main["Marker"] == mk)]
        if not sel.empty:
            row = sel.iloc[0]
            break
    else:
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
        f"Representative marker — {row['Movement']} "
        f"({MOVEMENT_DESCRIPTION.get(row['Movement'], '')}) — "
        f"{row['Marker']} — {row['File']}")
    plt.tight_layout()
    plt.savefig(OUT / "Fig09_Example_Norms.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def fig_stability(df_stability, stab_series):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for s in stab_series.values():
        tt = np.arange(len(s["Trans_norm"])) / s["fs"]
        axes[0].plot(tt, s["Trans_norm"], alpha=0.5)
        axes[1].plot(tt, s["Rot"], alpha=0.5)
    for tol in TRANSLATION_TOLERANCES_MM:
        axes[0].axhline(tol, color="red", linestyle="--", linewidth=1,
                        label=f"{tol} mm")
    axes[1].axhline(ROTATION_TOLERANCE_DEG, color="red", linestyle="--",
                    linewidth=1, label=f"{ROTATION_TOLERANCE_DEG}°")
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
    plt.savefig(OUT / "Fig10_Headband_Drift_Series.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    movements = df_stability["Movement"].unique()
    order_mv = sorted(movements)
    for ax, c, ylabel in [
        (axes[0], "Trans_norm_max_drift_mm", "Max translation drift [mm]"),
        (axes[1], "Rot_drift_max_deg", "Max rotation drift [deg]"),
        (axes[2], "Rot_dev_rms_deg",
         "Rotation deviation about mean offset, RMS [deg]"),
    ]:
        sns.boxplot(data=df_stability, x="Movement", y=c, ax=ax,
                    order=order_mv, color="lightsteelblue")
        sns.stripplot(data=df_stability, x="Movement", y=c,
                      color="black", alpha=0.5, size=4, ax=ax,
                      order=order_mv)
        ax.set_ylabel(ylabel)
        _movement_axis_labels(ax, movements)
    for tol in TRANSLATION_TOLERANCES_MM:
        axes[0].axhline(tol, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(ROTATION_TOLERANCE_DEG, color="red",
                    linestyle="--", linewidth=1)
    fig.suptitle(f"{LABEL_HEADBAND} drift by movement type")
    plt.tight_layout()
    plt.savefig(OUT / "Fig11_Stability_by_Movement.png", dpi=300,
                bbox_inches="tight")
    plt.close()

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
                  df_skin, df_skin_fixed, df_noise,
                  df_trial_class, df_amp_details):

    df_all.to_csv(OUT / "01_Marqueurs_Comparaison.csv",
                  index=False, encoding="utf-8-sig")

    if df_noise is not None and not df_noise.empty:
        df_noise.to_csv(OUT / "01b_Variabilite_Initiale_Par_Marqueur.csv",
                        index=False, encoding="utf-8-sig")

    if df_trial_class is not None and not df_trial_class.empty:
        df_trial_class.to_csv(OUT / "01c_Classification_Par_Essai.csv",
                              index=False, encoding="utf-8-sig")
        df_trial_class.groupby("Movement").agg(
            N_trials=("File", "count"),
            N_facial_median=("N_facial", "median"),
            N_analyzed_median=("N_analyzed", "median"),
            N_excluded_trajectory_median=("N_excluded_trajectory", "median"),
            EarlyWindow_SD_face_median_mm=("EarlyWindow_SD_face_mm", "median"),
            Amp_threshold_median_mm=("Amp_threshold_mm", "median"),
            N_moving_median=("N_moving", "median"),
            N_moving_min=("N_moving", "min"),
            N_moving_max=("N_moving", "max"),
            N_static_median=("N_static", "median"),
            N_discordant_median=("N_discordant", "median"),
            N_unclassifiable_median=("N_unclassifiable", "median"),
        ).round(3).to_csv(
            OUT / "01d_Classification_Resume.csv", encoding="utf-8-sig")

    if df_amp_details is not None and not df_amp_details.empty:
        df_amp_details.to_csv(OUT / "01e_Amplitudes_Par_Marqueur.csv",
                              index=False, encoding="utf-8-sig")

    if not df_stability.empty:
        df_stability.to_csv(OUT / "02_Headband_Stabilite.csv",
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
        df_per_file.to_csv(OUT / "16_Par_Fichier_Tous_Marqueurs.csv",
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
        movement_rows = []
        for movement, d_all in df_main.groupby("Movement"):
            d_mov = d_all[d_all["Marker_class"] == "moving"]
            row = {"Movement": movement,
                   "N_all_analysable": int(len(d_all)),
                   "N_moving_secondary": int(len(d_mov))}
            for p in KIN:
                row[f"{p}_RMSE_median_all"] = d_all[col(p, "RMSE")].median()
                row[f"{p}_RMSE_norm_median_all"] = d_all[col(p, "RMSE_norm")].median()
                row[f"{p}_NRMSE_median_moving_pct"] = (
                    d_mov[f"{p}_NRMSE_pct"].median() if not d_mov.empty else np.nan)
                row[f"{p}_Peak_relerr_median_moving_pct"] = (
                    d_mov[f"{p}_Peak_relerr_pct"].median() if not d_mov.empty else np.nan)
                row[f"{p}_Angle_mean_moving_deg"] = (
                    d_mov[f"{p}_Angle_mean_deg"].mean() if not d_mov.empty else np.nan)
                row[f"{p}_Pearson_norm_median_moving"] = (
                    d_mov[f"{p}_Pearson_norm"].median() if not d_mov.empty else np.nan)
            movement_rows.append(row)
        pandas.DataFrame(movement_rows).round(4).to_csv(
            OUT / "03_Synthese_par_Mouvement.csv", index=False,
            encoding="utf-8-sig")

        marker_rows = []
        for marker, d_all in df_main.groupby("Marker"):
            d_mov = d_all[d_all["Marker_class"] == "moving"]
            row = {"Marker": marker,
                   "N_all_analysable": int(len(d_all)),
                   "N_moving_secondary": int(len(d_mov))}
            for p in KIN:
                row[f"{p}_RMSE_mean_all"] = d_all[col(p, "RMSE")].mean()
                row[f"{p}_RMSE_norm_mean_all"] = d_all[col(p, "RMSE_norm")].mean()
                row[f"{p}_NRMSE_mean_moving_pct"] = (
                    d_mov[f"{p}_NRMSE_pct"].mean() if not d_mov.empty else np.nan)
                row[f"{p}_Angle_mean_moving_deg"] = (
                    d_mov[f"{p}_Angle_mean_deg"].mean() if not d_mov.empty else np.nan)
            marker_rows.append(row)
        pandas.DataFrame(marker_rows).round(4).to_csv(
            OUT / "04_Synthese_par_Marqueur.csv", index=False,
            encoding="utf-8-sig")

    groups = {
        "all_analysable_primary": df_main,
        "moving_secondary": df_main[df_main["Marker_class"] == "moving"],
        "static_secondary": df_main[df_main["Marker_class"] == "static"],
        "discordant_secondary": df_main[df_main["Marker_class"] == "discordant"],
        "unclassifiable_secondary": df_main[
            df_main["Marker_class"] == "unclassifiable"],
        "moving_QC_ok_secondary": df_main[
            (df_main["Marker_class"] == "moving") & (df_main["QC"] == "ok")],
    }
    rows = []
    for gname, d in groups.items():
        if d.empty:
            continue
        for p in KIN:
            r = {"Group": gname, "Quantity": p, "Unit": unit_label(p),
                 "N": len(d)}
            for label, fn, relative in REL_METRICS:
                if relative and gname != "moving_secondary":
                    r[label] = "n/a (métrique relative réservée au sous-groupe moving)"
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
            x = df_static[["Subject", cg, cd]].dropna()
            ratio = (x[cd] / x[cg]).replace([np.inf, -np.inf], np.nan)
            subject_x = x.groupby("Subject")[[cg, cd]].median().dropna()
            try:
                pv = (wilcoxon(subject_x[cg], subject_x[cd])[1]
                      if len(subject_x) >= 6
                      and (subject_x[cg] - subject_x[cd]).abs().sum() > 0
                      else np.nan)
            except Exception:
                pv = np.nan
            rows.append({
                "Metric": label, "Unit": unit, "N_marker_trial_pairs": len(x),
                "N_subjects_in_test": len(subject_x),
                "Gutter_median_IQR": med_iqr(x[cg]),
                "Headband_median_IQR": med_iqr(x[cd]),
                "Ratio_D_over_G_median": round(float(ratio.median()), 3),
                "Wilcoxon_p_subject_level": pv,
                "Interpretation": "exploratory low-amplitude subgroup; not a dedicated static noise calibration",
            })
        pandas.DataFrame(rows).to_csv(
            OUT / "10_Variabilite_Marqueurs_Faible_Amplitude.csv",
            index=False, encoding="utf-8-sig")

        error_rows = []
        for p in KIN:
            error_rows.append({
                "Quantity": p, "Unit": unit_label(p),
                "N_marker_trial_pairs": int(len(df_static)),
                "N_subjects": int(df_static["Subject"].nunique()),
                "Vector_RMSE_median_IQR": med_iqr(df_static[col(p, "RMSE")]),
                "Norm_RMSE_median_IQR": med_iqr(df_static[col(p, "RMSE_norm")]),
                "Interpretation": "pairwise disagreement within low-amplitude subgroup; not a device noise estimate",
            })
        pandas.DataFrame(error_rows).to_csv(
            OUT / "10b_Erreur_Cinematique_SousGroupe_FaibleAmplitude.csv",
            index=False, encoding="utf-8-sig")

    if not df_sweep.empty:
        d = df_sweep[df_sweep["Marker_class"] == "moving"].copy()
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
# ANALYSE ANATOMIQUE — CARTOGRAPHIE DES 105 MARQUEURS
# =====================================================================

def load_zone_mapping(path):
    """Lit le CSV Zones/Marqueurs (une zone par ligne, marqueurs en colonnes)."""
    if not Path(path).exists():
        print(f"[WARN] Cartographie anatomique absente : {path}")
        return {}, pandas.DataFrame()
    raw = pandas.read_csv(path, encoding="utf-8-sig", dtype=str)
    if raw.shape[1] < 2 or raw.columns[0].strip().lower() not in {"zones", "zone"}:
        raise ValueError("Format inattendu du CSV zones : première colonne attendue = 'Zones'.")
    mapping = {}
    duplicates = []
    for _, row in raw.iterrows():
        zone = str(row.iloc[0]).strip()
        if not zone or zone.lower() == "nan":
            continue
        for value in row.iloc[1:]:
            if pandas.isna(value):
                continue
            marker = str(value).strip()
            if not marker:
                continue
            if marker in mapping:
                duplicates.append(marker)
            mapping[marker] = zone
    if duplicates:
        raise ValueError("Marqueurs présents dans plusieurs zones : " + ", ".join(sorted(set(duplicates))))
    return mapping, raw


def zone_side(zone):
    z = str(zone)
    if "Droit" in z or "Droit " in z:
        return "Right"
    if "Gauche" in z:
        return "Left"
    return "Midline_or_unspecified"


def zone_pair_name(zone):
    z = str(zone)
    z = z.replace("Droit", "SIDE").replace("Gauche", "SIDE")
    return z.replace("SIDE", "Paired_region")


def _zone_summary_frame(df, group_cols, output_path, moving_only=False):
    if df is None or df.empty:
        return
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "N_marker_trial_pairs": int(len(g)),
            "N_markers": int(g["Marker"].nunique()),
            "N_subjects": int(g["Subject"].nunique()),
            "N_trials": int(g["File"].nunique()),
        })
        for p in KIN:
            row[f"{p}_Vector_RMSE_median"] = g[col(p, "RMSE")].median()
            row[f"{p}_Norm_RMSE_median"] = g[col(p, "RMSE_norm")].median()
            gm = g[g["Marker_class"] == "moving"]
            row[f"{p}_N_moving_pairs"] = int(len(gm))
            row[f"{p}_NRMSE_median_moving_pct"] = gm[f"{p}_NRMSE_pct"].median() if not gm.empty else np.nan
            row[f"{p}_Pearson_median_moving"] = gm[f"{p}_Pearson_norm"].median() if not gm.empty else np.nan
            row[f"{p}_Abs_peak_relerr_median_moving_pct"] = gm[f"{p}_Peak_relerr_pct"].abs().median() if not gm.empty else np.nan
        rows.append(row)
    pandas.DataFrame(rows).round(5).to_csv(OUT / output_path, index=False, encoding="utf-8-sig")


def export_anatomical_tables(df_main, df_amp_details, df_skipped, zone_mapping):
    """Exports anatomiques descriptifs. Pas de p-values au niveau marqueur."""
    if not zone_mapping:
        print("[WARN] Analyses anatomiques ignorées : cartographie non chargée.")
        return

    # Contrôle de la cartographie et de sa couverture dans les données observées.
    expected = pandas.DataFrame({"Marker": list(zone_mapping.keys()), "Zone": list(zone_mapping.values())})
    expected["Side"] = expected["Zone"].map(zone_side)
    expected["Zone_pair"] = expected["Zone"].map(zone_pair_name)
    observed = set(df_main["Marker"].dropna().astype(str)) if not df_main.empty else set()
    expected["Observed_in_analysis"] = expected["Marker"].isin(observed)
    expected["N_analysed"] = expected["Marker"].map(df_main.groupby("Marker").size() if not df_main.empty else {}).fillna(0).astype(int)
    expected["N_subjects"] = expected["Marker"].map(df_main.groupby("Marker")["Subject"].nunique() if not df_main.empty else {}).fillna(0).astype(int)
    expected["N_excluded_trajectory"] = expected["Marker"].map(df_skipped.groupby("Marker").size() if df_skipped is not None and not df_skipped.empty else {}).fillna(0).astype(int)
    expected.to_csv(OUT / "25_Couverture_Anatomique_105_Marqueurs.csv", index=False, encoding="utf-8-sig")

    if not df_main.empty:
        d = df_main.copy()
        d["Zone"] = d["Marker"].map(zone_mapping).fillna("UNMAPPED")
        d["Side"] = d["Zone"].map(zone_side)
        d["Zone_pair"] = d["Zone"].map(zone_pair_name)
        d.to_csv(OUT / "27_Marqueurs_Comparaison_Avec_Zones.csv", index=False, encoding="utf-8-sig")
        _zone_summary_frame(d, ["Zone"], "21_Analyse_Par_Zone.csv")
        _zone_summary_frame(d, ["Zone", "Movement"], "24_Analyse_Zone_Mouvement.csv")
        _zone_summary_frame(d, ["Subject", "Zone", "Movement"], "26_Analyse_Participant_Zone_Mouvement.csv")

        # Comparaison droite/gauche descriptive, uniquement pour les zones nommées Droit/Gauche.
        paired = d[d["Side"].isin(["Right", "Left"])].copy()
        if not paired.empty:
            _zone_summary_frame(paired, ["Zone_pair", "Side", "Movement"], "23_Comparaison_Droite_Gauche.csv")

        # Résumé couverture par zone
        cov = expected.groupby("Zone").agg(
            N_markers_expected=("Marker", "nunique"),
            N_markers_observed=("Observed_in_analysis", "sum"),
            N_marker_trial_pairs=("N_analysed", "sum"),
            N_excluded_trajectory=("N_excluded_trajectory", "sum"),
        ).reset_index()
        cov.to_csv(OUT / "25b_Couverture_Par_Zone.csv", index=False, encoding="utf-8-sig")

    if df_amp_details is not None and not df_amp_details.empty:
        a = df_amp_details.copy()
        a["Zone"] = a["Marker"].map(zone_mapping).fillna("UNMAPPED")
        counts = (a.groupby(["Movement", "Zone", "Class"], dropna=False)
                  .size().rename("N_marker_trial_pairs").reset_index())
        totals = a.groupby(["Movement", "Zone"]).size().rename("N_total_pairs").reset_index()
        counts = counts.merge(totals, on=["Movement", "Zone"], how="left")
        counts["Percentage_within_zone_movement"] = 100 * counts["N_marker_trial_pairs"] / counts["N_total_pairs"]
        counts.to_csv(OUT / "22_Discordances_Par_Zone_Mouvement.csv", index=False, encoding="utf-8-sig")
        discord = counts[counts["Class"] == "discordant"].copy()
        discord.to_csv(OUT / "22b_Taux_Discordance_Par_Zone.csv", index=False, encoding="utf-8-sig")


def fig_anatomical_heatmap(df_main, zone_mapping):
    if not zone_mapping or df_main is None or df_main.empty:
        return
    d = df_main[df_main["Marker_class"] == "moving"].copy()
    if d.empty:
        return
    d["Zone"] = d["Marker"].map(zone_mapping)
    for p in KIN:
        c = f"{p}_NRMSE_pct"
        tab = d.pivot_table(index="Zone", columns="Movement", values=c, aggfunc="median")
        if tab.empty:
            continue
        h = max(7, 0.28 * len(tab))
        fig, ax = plt.subplots(figsize=(10, h))
        sns.heatmap(tab, annot=True, fmt=".1f", cmap="viridis_r", linewidths=.25,
                    cbar_kws={"label": "Median NRMSE (%)"}, ax=ax)
        ax.set_title(f"{p} — median NRMSE by anatomical zone and movement (moving markers)")
        ax.set_xlabel("Movement")
        ax.set_ylabel("Anatomical zone")
        plt.tight_layout()
        plt.savefig(OUT / f"Fig13_{p}_NRMSE_Zone_Mouvement.png", dpi=300, bbox_inches="tight")
        plt.close()


# =====================================================================
# PROGRAMME PRINCIPAL
# =====================================================================


# =====================================================================
# V12.4 — RELATION AMPLITUDE / ERREUR / VARIABILITE
# =====================================================================
def _safe_spearman(x, y):
    z = pandas.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 5 or z.x.nunique() < 3 or z.y.nunique() < 3:
        return np.nan, np.nan, len(z)
    r, p = spearmanr(z.x, z.y)
    return float(r), float(p), len(z)


def amplitude_error_analysis(df_main):
    """Analyse exploratoire amplitude-erreur. Les inférences rééchantillonnent les sujets."""
    if df_main.empty:
        return
    d = df_main.copy()
    # Variables harmonisées pour chaque grandeur.
    specs = {
        "Pos": ("Pos_Amplitude_mean", "Pos_RMSE_mm", "Pos_NRMSE_pct", "mm"),
        "Vel": ("Vel_Amplitude_mean", "Vel_RMSE_mm_s", "Vel_NRMSE_pct", "mm/s"),
        "Acc": ("Acc_Amplitude_mean", "Acc_RMSE_mm_s2", "Acc_NRMSE_pct", "mm/s2"),
    }
    rows = []
    for q, (amp, rmse, nrmse, unit) in specs.items():
        if not all(c in d.columns for c in (amp, rmse, nrmse)):
            continue
        z = d[["File", "Subject", "Movement", "Marker", "Marker_class", amp, rmse, nrmse]].copy()
        z = z.rename(columns={amp:"Amplitude", rmse:"RMSE_absolute", nrmse:"NRMSE_pct"})
        z["Quantity"] = q
        z["Amplitude_unit"] = unit
        z["Relative_error_to_amplitude_pct"] = 100*z.RMSE_absolute/z.Amplitude.replace(0,np.nan)
        rows.append(z)
    if not rows:
        return
    long = pandas.concat(rows, ignore_index=True)
    long.replace([np.inf,-np.inf],np.nan).to_csv(OUT/"28_Amplitude_Erreur_Global.csv",index=False,encoding="utf-8-sig")

    summaries=[]
    for (q,mov), g in long.groupby(["Quantity","Movement"]):
        for subset_name, h in [("all_analysable",g),("moving_secondary",g[g.Marker_class=="moving"])]:
            rho,pv,n=_safe_spearman(h.Amplitude,h.NRMSE_pct)
            rho_abs,pv_abs,n_abs=_safe_spearman(h.Amplitude,h.RMSE_absolute)
            summaries.append({"Quantity":q,"Movement":mov,"Subset":subset_name,"N_marker_trials":n,
                "N_subjects":h.Subject.nunique(),"Amplitude_median":h.Amplitude.median(),
                "RMSE_absolute_median":h.RMSE_absolute.median(),"NRMSE_median_pct":h.NRMSE_pct.median(),
                "Spearman_amplitude_NRMSE":rho,"Spearman_p_descriptive_only":pv,
                "Spearman_amplitude_absolute_RMSE":rho_abs,"Spearman_abs_RMSE_p_descriptive_only":pv_abs})
    pandas.DataFrame(summaries).to_csv(OUT/"29_Amplitude_Erreur_Par_Mouvement.csv",index=False,encoding="utf-8-sig")

    # Agrégation à poids égal par participant × mouvement × marqueur, puis participant.
    participant = (long.groupby(["Quantity","Subject","Movement"],as_index=False)
                   .agg(N_marker_trials=("Amplitude","size"),Amplitude_median=("Amplitude","median"),
                        RMSE_absolute_median=("RMSE_absolute","median"),NRMSE_median_pct=("NRMSE_pct","median")))
    participant.to_csv(OUT/"30_Amplitude_Erreur_Par_Participant.csv",index=False,encoding="utf-8-sig")

    # Proxy de variabilité mesuré dans les mêmes trajectoires, pas bruit instrumental pur.
    noise_cols=[c for c in ["G_Pos_SD_mm","D_Pos_SD_mm","G_Vel_RMS_mm_s","D_Vel_RMS_mm_s","G_Acc_RMS_mm_s2","D_Acc_RMS_mm_s2"] if c in d.columns]
    if noise_cols:
        noise=d[[c for c in ["File","Subject","Movement","Marker","Marker_class"] if c in d.columns]+noise_cols].copy()
        noise.to_csv(OUT/"31_Variabilite_Dispositifs.csv",index=False,encoding="utf-8-sig")

    # V12.4.1 — analyse intra-participant ET intra-mouvement.
    # Chaque corrélation est calculée au sein d'un groupe Subject × Movement,
    # ce qui évite de confondre l'effet amplitude avec les différences entre mouvements
    # ou entre participants. Les groupes sont ensuite agrégés à poids égal par sujet.
    # Avec 9 sujets, les IC bootstrap sont descriptifs/prudents ; pas de p-values GEE
    # interprétées comme confirmatoires.
    within_rows=[]
    for q,g in long.groupby("Quantity"):
        for (sub,mov), h0 in g.groupby(["Subject","Movement"]):
            h=h0[(h0.Amplitude>0)&(h0.RMSE_absolute>0)&(h0.NRMSE_pct>0)].copy()
            for outcome, col in [("NRMSE", "NRMSE_pct"),("RMSE_absolute","RMSE_absolute")]:
                rho,pv,n=_safe_spearman(h.Amplitude,h[col])
                within_rows.append({"Quantity":q,"Subject":sub,"Movement":mov,"Outcome":outcome,
                    "N_marker_trials":n,"Amplitude_median":h.Amplitude.median(),
                    "Outcome_median":h[col].median(),"Spearman_rho":rho,
                    "p_descriptive_only":pv,
                    "Eligible_for_group_summary":bool(n>=5 and h.Amplitude.nunique()>=3 and h[col].nunique()>=3)})
    within=pandas.DataFrame(within_rows)
    within.to_csv(OUT/"32_Analyse_IntraParticipant_Mouvement.csv",index=False,encoding="utf-8-sig")

    # Médiane des corrélations mouvement au sein de chaque sujet : poids égal par sujet.
    subject_rows=[]
    eligible=within[within.Eligible_for_group_summary].copy() if not within.empty else within
    if not eligible.empty:
        for (q,sub,outcome),h in eligible.groupby(["Quantity","Subject","Outcome"]):
            subject_rows.append({"Quantity":q,"Subject":sub,"Outcome":outcome,
                "N_movements":h.Movement.nunique(),"Spearman_median_across_movements":h.Spearman_rho.median(),
                "Spearman_min":h.Spearman_rho.min(),"Spearman_max":h.Spearman_rho.max()})
    subj_summary=pandas.DataFrame(subject_rows)
    subj_summary.to_csv(OUT/"33_Synthese_IntraParticipant.csv",index=False,encoding="utf-8-sig")

    # Bootstrap en grappes : rééchantillonnage des participants (et non des marqueurs).
    rng=np.random.default_rng(20261113)
    boot_rows=[]
    if not subj_summary.empty:
        for (q,outcome),h in subj_summary.groupby(["Quantity","Outcome"]):
            vals=h.Spearman_median_across_movements.dropna().to_numpy()
            boots=[]
            if len(vals)>=4:
                for _ in range(5000):
                    sample=rng.choice(vals,size=len(vals),replace=True)
                    boots.append(float(np.median(sample)))
            boot_rows.append({"Quantity":q,"Outcome":outcome,"N_subjects":len(vals),
                "Median_subject_correlation":float(np.median(vals)) if len(vals) else np.nan,
                "Bootstrap_CI95_low":float(np.percentile(boots,2.5)) if boots else np.nan,
                "Bootstrap_CI95_high":float(np.percentile(boots,97.5)) if boots else np.nan,
                "N_bootstrap":len(boots),
                "Method":"Equal-weight subject medians; within-subject × movement Spearman; participant-cluster bootstrap; descriptive with n=9"})
    pandas.DataFrame(boot_rows).to_csv(OUT/"34_Bootstrap_IntraParticipant.csv",index=False,encoding="utf-8-sig")

    # Analyse complémentaire par centrage dans chaque groupe Subject × Movement.
    # Les coefficients sont descriptifs : aucune p-value n'est produite.
    centered_rows=[]
    for q,g in long.groupby("Quantity"):
        for outcome,col in [("NRMSE","NRMSE_pct"),("RMSE_absolute","RMSE_absolute")]:
            h=g[(g.Amplitude>0)&(g[col]>0)].copy()
            h["log_amp"]=np.log(h.Amplitude)
            h["log_outcome"]=np.log(h[col])
            h["x_centered"]=h.log_amp-h.groupby(["Subject","Movement"]).log_amp.transform("mean")
            h["y_centered"]=h.log_outcome-h.groupby(["Subject","Movement"]).log_outcome.transform("mean")
            den=float((h.x_centered**2).sum())
            beta=float((h.x_centered*h.y_centered).sum()/den) if den>0 else np.nan
            centered_rows.append({"Quantity":q,"Outcome":outcome,"Within_group_log_slope":beta,
                "N_marker_trials":len(h),"N_subjects":h.Subject.nunique(),"N_subject_movement_groups":h.groupby(["Subject","Movement"]).ngroups,
                "Interpretation":"Descriptive within Subject × Movement centered log-log slope; no confirmatory p-value"})
    pandas.DataFrame(centered_rows).to_csv(OUT/"35_Modele_Centre_IntraGroupe.csv",index=False,encoding="utf-8-sig")

    # Figures : tous les marqueurs, couleurs par mouvement, axes log si pertinents.
    for q,g in long.groupby("Quantity"):
        h=g[(g.Amplitude>0)&(g.NRMSE_pct>0)]
        if h.empty: continue
        fig,ax=plt.subplots(figsize=(9,6))
        sns.scatterplot(data=h,x="Amplitude",y="NRMSE_pct",hue="Movement",style="Movement",alpha=.55,s=35,ax=ax)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{q} — amplitude vs erreur relative (tous marqueurs analysables)")
        ax.set_xlabel(f"Amplitude du signal moyen G/D ({h.Amplitude_unit.iloc[0]}) — échelle log")
        ax.set_ylabel("NRMSE (%) — échelle log")
        ax.grid(True,which="both",alpha=.2); fig.tight_layout()
        fig.savefig(OUT/f"Fig14_{q}_Amplitude_vs_NRMSE.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,6))
        sns.scatterplot(data=h,x="Amplitude",y="RMSE_absolute",hue="Movement",alpha=.55,s=35,ax=ax)
        ax.set_xscale("log"); ax.set_title(f"{q} — amplitude vs erreur absolue")
        ax.set_xlabel(f"Amplitude ({h.Amplitude_unit.iloc[0]}) — échelle log")
        ax.set_ylabel(f"RMSE absolue ({h.Amplitude_unit.iloc[0]})")
        ax.grid(True,which="both",alpha=.2); fig.tight_layout()
        fig.savefig(OUT/f"Fig15_{q}_Amplitude_vs_RMSE_absolue.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def main():

    OUT.mkdir(parents=True, exist_ok=True)
    zone_mapping, zone_table = load_zone_mapping(ZONE_FILE)
    if zone_mapping:
        print(f"Cartographie anatomique chargée : {len(zone_mapping)} marqueurs / {len(zone_table)} zones ({ZONE_FILE.name})")
        if len(zone_mapping) != 105:
            print(f"[WARN] La cartographie contient {len(zone_mapping)} marqueurs, 105 attendus.")
    else:
        print("[WARN] Pas de cartographie anatomique : les analyses globales seront tout de même produites.")

    series_store = {}
    stab_series = {}
    all_markers, all_stability, all_sweep, all_skipped = [], [], [], []
    all_rigidity = []
    all_skin = []
    all_skin_fixed = []
    all_noise = []
    all_trial_class = []
    all_amp_details = []

    gutter_files = sorted(DIR_G.glob("*.csv"))

    print("\n" + "=" * 80)
    print("VALIDATION D'INTERCHANGEABILITÉ — SPLINT vs HEADBAND — V12.4")
    print("=" * 80)
    print(f"Fichiers splint trouvés : {len(gutter_files)}")
    print(f"Dossier de sortie : {OUT}")
    print(f"Repère : {REPERE_MODE} | Baseline : {BASELINE_FRAMES} frames")
    print(f"Filtre : Butterworth {FILTER_CUTOFF_HZ} Hz, ordre {FILTER_ORDER}")
    print(f"Balayage de coupure : {SWEEP_CUTOFFS_HZ} Hz")
    print(f"Seuil fixes : {FIXED_THRESHOLD_MM} mm | "
          f"MIN_FIXED_MARKERS : {MIN_FIXED_MARKERS}")
    print("Analyse principale : tous les marqueurs faciaux avec trajectoires "
          "exploitables, indépendamment de leur amplitude")
    print(f"Classification secondaire moving : amp > "
          f"max({AMP_MOVING_FACTOR}×variabilité initiale (proxy), "
          f"{AMP_MOVING_FLOOR_MM} mm), dans G ET D")
    print(f"Marge d'équivalence ROM (TOST) : ±{EQUIVALENCE_DELTA_MM} mm")
    print(f"Diamètres marqueurs — Gouttière : Ø5 mm | "
          f"Serre-tête : Ø14 mm | Face : Ø{FACE_DIAMETER_MM} mm")
    print(f"Mouvements attendus : "
          f"{', '.join(sorted(MOVEMENT_DESCRIPTION.keys()))}")

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

        missing = [m for m in REF_POINTS_G
                   if m not in gutter_data or m not in diadem_data]
        if missing:
            print(f"   ⚠ Repère incomplet : {missing}")
            continue

        facial_common = [m for m in gutter_data
                         if m in diadem_data
                         and m not in REF_POINTS_G
                         and m not in REF_POINTS_D]
        all_noise.extend(
            measure_early_window_variability(gutter_data, diadem_data, facial_common))

        (mres, sres, wres, skp, rres, kres, kfixed, tclass, ampdet) = \
            analyze_file_pair(
                gutter_data, diadem_data, gutter_file.name, fs_g,
                series_store, stab_series)

        all_markers.extend(mres)
        all_stability.extend(sres)
        all_sweep.extend(wres)
        all_skipped.extend(skp)
        all_rigidity.extend(rres)
        all_skin.extend(kres)
        all_skin_fixed.extend(kfixed)
        all_trial_class.extend(tclass)
        all_amp_details.extend(ampdet)

        n_mov = (len([x for x in tclass[0]["Moving_markers"].split("|") if x])
                 if tclass else 0)
        n_fix = 0
        if kfixed and kfixed[0]["Fixed_markers"]:
            n_fix = len(kfixed[0]["Fixed_markers"].split("|"))
        print(f"   ✓ {len(mres)}/{len(facial_common)} marqueurs analysés, "
              f"{len(skp)} exclus pour trajectoire insuffisante | "
              f"mobiles (secondaire): {n_mov} | fixes cutanés: {n_fix}")

    df_all = pandas.DataFrame(all_markers)
    df_stability = pandas.DataFrame(all_stability)
    df_sweep = pandas.DataFrame(all_sweep)
    df_skipped = pandas.DataFrame(all_skipped)
    df_rigidity = pandas.DataFrame(all_rigidity)
    df_skin = pandas.DataFrame(all_skin)
    df_skin_fixed = pandas.DataFrame(all_skin_fixed)
    df_noise = pandas.DataFrame(all_noise)
    df_trial_class = pandas.DataFrame(all_trial_class)
    df_amp_details = pandas.DataFrame(all_amp_details)

    print("\n" + "=" * 80)
    print(f"Résultats marqueurs   : {len(df_all)}")
    print(f"Résultats stabilité   : {len(df_stability)}")
    print(f"Résultats rigidité    : {len(df_rigidity)}")
    print(f"Résultats cutanés     : {len(df_skin)}")
    print(f"Variabilité fenêtre initiale : {len(df_noise)}")
    print(f"Classifications essai : {len(df_trial_class)}")

    if df_all.empty:
        print("Aucun résultat marqueur : arrêt.")
        return

    df_main = main_subset(df_all)
    df_static = df_main[df_main["Marker_class"] == "static"]
    df_sweep_main = (df_sweep[df_sweep["Marker_class"] == "moving"].copy()
                     if not df_sweep.empty else df_sweep)
    if USE_QC_FOR_MAIN and not df_sweep_main.empty:
        df_sweep_main = df_sweep_main[df_sweep_main["QC"] == "ok"]
    df_moving_main = df_main[df_main["Marker_class"] == "moving"].copy()

    test_cols = []
    for p in KIN:
        # Export par fichier de grandeurs absolues uniquement ; les métriques
        # relatives sont réservées au sous-groupe moving (voir synthèse 06).
        test_cols += [col(p, "Peak_G"), col(p, "Peak_D")]
    df_per_file = aggregate_per_file(df_main, test_cols)

    static_cols = ["G_Pos_SD_mm", "D_Pos_SD_mm",
                   "G_Vel_RMS_mm_s", "D_Vel_RMS_mm_s",
                   "G_Acc_RMS_mm_s2", "D_Acc_RMS_mm_s2"]
    df_static_per_file = (aggregate_per_file(df_static, static_cols)
                          if not df_static.empty else pandas.DataFrame())

    print(f"   analyse principale (tous marqueurs exploitables) : {len(df_main)}")
    print(f"   sous-groupe statique descriptif : {len(df_static)}")
    print(f"   Essais (fichiers) : {len(df_per_file)}")

    export_tables(df_all, df_main, df_static, df_stability, df_sweep,
                  df_skipped, df_rigidity, df_per_file, df_static_per_file,
                  df_skin, df_skin_fixed, df_noise,
                  df_trial_class, df_amp_details)

    # Analyses anatomiques descriptives (cartographie officielle 105 marqueurs).
    export_anatomical_tables(df_main, df_amp_details, df_skipped, zone_mapping)
    print("\n[V12.4] Analyse amplitude / erreur / variabilité...")
    amplitude_error_analysis(df_main)
    if zone_mapping:
        fig_anatomical_heatmap(df_main, zone_mapping)

    print("\n" + "=" * 80)
    print("GÉNÉRATION DES FIGURES (ordre = chaîne de validation)")
    print("=" * 80)

    if not df_rigidity.empty:
        print("   → Fig01_Rigidity_Diagnostics.png   [1/5]")
        fig_rigidity(df_rigidity)
    if not df_skin.empty:
        print("   → Fig02_Skin_Drift_Comparison.png  [2/5]")
        fig_skin_drift(df_skin)
    if not df_static.empty:
        print("   → Fig03_FaibleAmplitude_Variabilite.png    [3/5]")
        fig_static_noise(df_static)
    if not df_noise.empty or not df_amp_details.empty:
        print("   → Fig07_Variabilite_Initiale_Amplitude.png    [annexe]")
        fig_noise_and_amplitude(df_noise, df_amp_details)
    if not df_main.empty:
        print("   → Fig04_BlandAltman_ROM.png        [4/5]")
        fig_bland_altman_rom(df_main)
        print("   → Fig05_NRMSE_by_Movement.png      [4b]")
        fig_nrmse_by_movement(df_main)
    if not df_sweep_main.empty:
        print("   → Fig06_Peaks_vs_Cutoff.png        [5/5]")
        fig_cutoff_sweep(df_sweep_main)

    if not df_main.empty:
        print("   → Fig08_BlandAltman_Peaks.png      [annexe]")
        fig_bland_altman_peaks(df_moving_main)
        print("   → Fig09_Example_Norms.png          [annexe]")
        fig_example(df_main, series_store)
    if not df_stability.empty:
        print("   → Fig10_Headband_Drift_Series.png  [annexe]")
        print("   → Fig11_Stability_by_Movement.png  [annexe]")
        print("   → Fig12_Stability_by_Subject.png   [annexe]")
        fig_stability(df_stability, stab_series)

    # --- Résumé console ------------------------------------------
    print("\n" + "=" * 80)
    print("RÉSUMÉ — classification par mouvement")
    print("=" * 80)
    if not df_trial_class.empty:
        res = df_trial_class.groupby("Movement").agg(
            N=("File", "count"),
            N_moving_med=("N_moving", "median"),
            N_static_med=("N_static", "median"),
            N_discordant_med=("N_discordant", "median"),
            N_unclassifiable_med=("N_unclassifiable", "median"),
            Threshold_med=("Amp_threshold_mm", "median"),
        )
        for mv, row in res.iterrows():
            desc = MOVEMENT_DESCRIPTION.get(mv, "")
            print(f"   {mv} ({desc}) : n={int(row['N'])}, "
                  f"mobiles={row['N_moving_med']:.0f}, "
                  f"statiques={row['N_static_med']:.0f}, "
                  f"discordants={row['N_discordant_med']:.0f}, "
                  f"non classifiables={row['N_unclassifiable_med']:.0f}, "
                  f"seuil={row['Threshold_med']:.2f} mm")

    print("\n" + "=" * 80)
    print("RÉSUMÉ — marqueurs analysables et sous-groupe mobile secondaire")
    print("=" * 80)
    df_moving_secondary = df_main[df_main["Relative_metrics_interpretable"]]
    for p in KIN:
        if df_main.empty:
            break
        print(f"\n{p} [{unit_label(p)}]")
        print(f"   RMSE vectoriel (tous marqueurs) : "
              f"{df_main[col(p, 'RMSE')].median():.3f}")
        if not df_moving_secondary.empty:
            print(f"   NRMSE (sous-groupe moving)      : "
                  f"{df_moving_secondary[f'{p}_NRMSE_pct'].median():.1f}")
            print(f"   Pearson (sous-groupe moving)    : "
                  f"{df_moving_secondary[f'{p}_Pearson_norm'].median():.3f}")

    if not df_skin.empty:
        print(f"\nComparaison exploratoire au repère cutané (n={len(df_skin)} sessions) :")
        print(f"   Skin RMSD médian : "
              f"{df_skin['Skin_RMSD_mean_mm'].median():.3f} mm")
        print(f"   Diadème translation max : "
              f"{df_skin['D_Trans_norm_max_mm'].median():.2f} mm")
        print(f"   Splint  translation max : "
              f"{df_skin['G_Trans_norm_max_mm'].median():.2f} mm")
        print(f"   Ratio D/G translation   : "
              f"{df_skin['Trans_ratio_D_over_G'].median():.2f}")
        print(f"   Ratio D/G rotation      : "
              f"{df_skin['Rot_ratio_D_over_G'].median():.2f}")

    if not df_noise.empty:
        print("\nVariabilité de la fenêtre initiale (proxy exploratoire, mm) :")
        for role in ["Reference", "Facial"]:
            sub = df_noise[df_noise["Role"] == role]
            if sub.empty:
                continue
            for dev in ["Gutter", "Headband"]:
                s = sub[sub["Device"] == dev]["EarlyWindow_SD_mm"].dropna()
                if not s.empty:
                    print(f"   {role:9s} {dev:9s} : "
                          f"{s.median():.3f} (n={len(s)})")

    print(f"\n✓ Analyse V12.3 terminée. Résultats dans :\n  {OUT}\n")


if __name__ == "__main__":
    main()