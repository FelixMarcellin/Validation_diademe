# -*- coding: utf-8 -*-
"""
======================================================================
COMPARAISON GOUTTIÈRE vs DIADEME
======================================================================

Objectifs
---------
1. Comparer les trajectoires des marqueurs faciaux enregistrés
   simultanément avec :
       - repère maxillaire / gouttière
       - repère diadème

2. Les acquisitions étant parfaitement synchronisées :
       -> comparaison frame par frame
       -> aucun recalage temporel

3. Comparaison des marqueurs faciaux dans le REPÈRE GOUTTIÈRE.

4. Analyse de la stabilité du diadème dans le repère gouttière :
       - translation
       - rotation
       - drift X/Y/Z
       - drift 3D
       - variation de rotation

5. Génération :
       - figures
       - tableaux CSV
       - statistiques (Kruskal-Wallis + Friedman)
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import os
from pathlib import Path

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, kruskal, friedmanchisquare


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
    r"\RESULTATS_COMPARAISON_V6"
)

OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Marqueurs définissant le repère local
# ---------------------------------------------------------------------

REF_POINTS = ["M1", "M2", "M3"]


# ---------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------

MIN_VALID_FRAMES = 10
EPS = 1e-9

# Seuil clinique de tolérance pour la translation du diadème (mm)
TRANSLATION_TOLERANCE_MM = 2.0

# Seuil clinique de tolérance pour la rotation du diadème (deg)
ROTATION_TOLERANCE_DEG = 2.0


# =====================================================================
# OUTILS
# =====================================================================

def safe_norm(vector):
    """Norme euclidienne d'un vecteur."""
    return np.linalg.norm(vector)


def rotation_angle_from_matrix(R):
    """
    Angle de rotation correspondant à une matrice de rotation 3x3.
    Retour en degrés.
    """
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    return np.degrees(np.arccos(value))


# =====================================================================
# CONSTRUCTION DU REPÈRE LOCAL
# =====================================================================

def get_local_basis(m1, m2, m3):
    """
    Construit un repère orthonormé à partir de M1, M2, M3.

    Origine :
        M1

    Axe X :
        M1 -> M2

    Axe Z :
        perpendiculaire au plan M1-M2-M3

    Axe Y :
        complète le trièdre orthonormé.

    Retour
    ------
    R : matrice 3x3
        Colonnes = axes du repère local exprimés dans le repère global.

    origin : coordonnées de M1
    """

    try:

        m1 = np.asarray(m1, dtype=float)
        m2 = np.asarray(m2, dtype=float)
        m3 = np.asarray(m3, dtype=float)

        if (
            np.isnan(m1).any()
            or np.isnan(m2).any()
            or np.isnan(m3).any()
        ):
            return None, None

        origin = m1.copy()

        # --------------------------------------------------------------
        # X
        # --------------------------------------------------------------

        vx = m2 - m1
        norm_x = np.linalg.norm(vx)

        if norm_x < EPS:
            return None, None

        vx = vx / norm_x

        # --------------------------------------------------------------
        # Z
        # --------------------------------------------------------------

        v_temp = m3 - m1

        vz = np.cross(vx, v_temp)
        norm_z = np.linalg.norm(vz)

        if norm_z < EPS:
            return None, None

        vz = vz / norm_z

        # --------------------------------------------------------------
        # Y
        # --------------------------------------------------------------

        vy = np.cross(vz, vx)

        norm_y = np.linalg.norm(vy)

        if norm_y < EPS:
            return None, None

        vy = vy / norm_y

        # --------------------------------------------------------------
        # Matrice
        # --------------------------------------------------------------

        R = np.column_stack((vx, vy, vz))

        return R, origin

    except Exception:
        return None, None


# =====================================================================
# LECTURE VICON
# =====================================================================

def load_vicon(file_path):
    """
    Lecture des CSV Vicon.

    La structure attendue est celle utilisée dans ton script original :
        - 4 lignes ignorées pour le DataFrame
        - coordonnées XYZ par blocs de 3 colonnes
        - noms de marqueurs récupérés dans la ligne 3.

    Retour
    ------
    dict :
        {
            "M1": ndarray(N,3),
            "M2": ndarray(N,3),
            ...
        }
    """

    try:

        # --------------------------------------------------------------
        # Lecture numérique
        # --------------------------------------------------------------

        dataframe = pandas.read_csv(
            file_path,
            skiprows=4,
            low_memory=False
        )

        dataframe = dataframe.apply(
            pandas.to_numeric,
            errors="coerce"
        )

        # --------------------------------------------------------------
        # Lecture des noms de marqueurs
        # --------------------------------------------------------------

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            lines = file.readlines()

        if len(lines) < 3:
            return None

        marker_names = [
            element.split(":")[-1].strip()
            for element in lines[2].strip().split(",")
            if element.strip()
        ]

        # --------------------------------------------------------------
        # Extraction XYZ
        # --------------------------------------------------------------

        data = {}

        for i, marker_name in enumerate(marker_names):

            if not marker_name:
                continue

            start_col = i * 3 + 2
            end_col = start_col + 3

            if end_col > dataframe.shape[1]:
                continue

            xyz = dataframe.iloc[:, start_col:end_col].values

            if xyz.shape[1] != 3:
                continue

            data[marker_name] = xyz.astype(float)

        if len(data) == 0:
            return None

        return data

    except Exception as error:

        print(
            f"[ERREUR LECTURE] {file_path.name} : {error}"
        )

        return None


# =====================================================================
# APPARIEMENT DES FICHIERS
# =====================================================================

def find_matching_diadem_file(gutter_file):
    """
    Recherche du fichier diadème correspondant.

    Priorité :
        1. remplacement explicite _G_ -> _D_
        2. remplacement G -> D si nécessaire
    """

    # --------------------------------------------------------------
    # Méthode principale
    # --------------------------------------------------------------

    candidate = DIR_D / gutter_file.name.replace("_G_", "_D_")

    if candidate.exists():
        return candidate

    # --------------------------------------------------------------
    # Recherche plus tolérante
    # --------------------------------------------------------------

    name = gutter_file.name

    variants = [
        name.replace("_G_", "_D_"),
        name.replace("_g_", "_d_"),
        name.replace("G_", "D_"),
        name.replace("_G", "_D"),
    ]

    for variant in variants:

        candidate = DIR_D / variant

        if candidate.exists():
            return candidate

    return None


# =====================================================================
# IDENTIFICATION DU MOUVEMENT
# =====================================================================

def extract_movement(file_name):
    """
    Extrait M1, M2, etc. à partir du nom de fichier.
    """

    name = Path(file_name).stem

    if "_M" not in name:
        return "M?"

    try:

        after_m = name.split("_M", 1)[1]

        number = ""

        for char in after_m:

            if char.isdigit():
                number += char
            else:
                break

        if number:
            return f"M{number}"

    except Exception:
        pass

    return "M?"


# =====================================================================
# IDENTIFICATION DU SUJET
# =====================================================================

def extract_subject(file_name):
    """
    Extrait un identifiant sujet à partir du nom de fichier.

    Format attendu : 20230614_G_SMA_M1.csv
                     -> renvoie 'SMA'

    Fallback : renvoie le nom de fichier complet.
    """
    try:
        name = Path(file_name).stem
        parts = name.split("_")
        # Cherche l'initiale du sujet (habituellement index 2)
        if len(parts) >= 3:
            return parts[2]
        return name
    except Exception:
        return file_name


# =====================================================================
# TRANSFORMATION D'UN POINT
# =====================================================================

def point_to_local(point_global, R, origin):
    """
    Transforme un point du repère global vers le repère local.
    """

    return R.T @ (point_global - origin)


# =====================================================================
# TRANSFORMATION DU REPÈRE DIADEME DANS LE REPÈRE GOUTTIÈRE
# =====================================================================

def relative_transform_gutter_to_diadem(
    R_g,
    origin_g,
    R_d,
    origin_d
):
    """
    Transformation du repère diadème exprimée dans le repère gouttière.

    Retour :
        translation :
            origine du diadème dans le repère gouttière

        rotation :
            orientation du diadème dans le repère gouttière
    """

    translation = R_g.T @ (origin_d - origin_g)

    rotation = R_g.T @ R_d

    return translation, rotation


# =====================================================================
# ANALYSE D'UN FICHIER
# =====================================================================

def analyze_file_pair(
    gutter_data,
    diadem_data,
    file_name
):

    movement = extract_movement(file_name)

    # --------------------------------------------------------------
    # Marqueurs communs
    # --------------------------------------------------------------

    common_markers = [
        marker
        for marker in gutter_data.keys()
        if marker in diadem_data
    ]

    facial_markers = [
        marker
        for marker in common_markers
        if marker not in REF_POINTS
    ]

    # --------------------------------------------------------------
    # Nombre de frames
    # --------------------------------------------------------------

    available_lengths = []

    for marker in REF_POINTS:

        if marker in gutter_data:
            available_lengths.append(len(gutter_data[marker]))

        if marker in diadem_data:
            available_lengths.append(len(diadem_data[marker]))

    if len(available_lengths) == 0:
        return [], []

    n_frames = min(available_lengths)

    # =================================================================
    # CONSTRUCTION DES REPÈRES
    # =================================================================

    gutter_frames = []
    diadem_frames = []

    for frame in range(n_frames):

        R_g, origin_g = get_local_basis(
            gutter_data["M1"][frame],
            gutter_data["M2"][frame],
            gutter_data["M3"][frame]
        )

        R_d, origin_d = get_local_basis(
            diadem_data["M1"][frame],
            diadem_data["M2"][frame],
            diadem_data["M3"][frame]
        )

        if (
            R_g is None
            or R_d is None
        ):
            gutter_frames.append(None)
            diadem_frames.append(None)

        else:

            gutter_frames.append(
                (R_g, origin_g)
            )

            diadem_frames.append(
                (R_d, origin_d)
            )

    # =================================================================
    # COMPARAISON DES MARQUEURS FACIAUX
    # =================================================================

    marker_results = []

    for marker in facial_markers:

        gutter_local = []
        diadem_in_gutter = []

        frame_indices = []

        for frame in range(n_frames):

            gutter_frame = gutter_frames[frame]
            diadem_frame = diadem_frames[frame]

            if (
                gutter_frame is None
                or diadem_frame is None
            ):
                continue

            R_g, origin_g = gutter_frame
            R_d, origin_d = diadem_frame

            # ----------------------------------------------------------
            # Marqueur dans repère gouttière
            # ----------------------------------------------------------

            point_g = point_to_local(
                gutter_data[marker][frame],
                R_g,
                origin_g
            )

            # ----------------------------------------------------------
            # Marqueur dans repère diadème
            # puis transformé dans le repère gouttière
            # ----------------------------------------------------------

            point_d_local = point_to_local(
                diadem_data[marker][frame],
                R_d,
                origin_d
            )

            point_d_in_g = (
                R_g.T @ R_d @ point_d_local
                + R_g.T @ (origin_d - origin_g)
            )

            gutter_local.append(point_g)
            diadem_in_gutter.append(point_d_in_g)
            frame_indices.append(frame)

        # --------------------------------------------------------------
        # Nombre de frames valides
        # --------------------------------------------------------------

        if len(gutter_local) < MIN_VALID_FRAMES:
            continue

        gutter_local = np.asarray(gutter_local)
        diadem_in_gutter = np.asarray(diadem_in_gutter)

        # ==============================================================
        # TRAJECTOIRES RELATIVES
        # ==============================================================

        gutter_relative = (
            gutter_local
            - gutter_local[0]
        )

        diadem_relative = (
            diadem_in_gutter
            - diadem_in_gutter[0]
        )

        # ==============================================================
        # ERREUR FRAME PAR FRAME
        # ==============================================================

        error_xyz = (
            diadem_relative
            - gutter_relative
        )

        error_norm = np.linalg.norm(
            error_xyz,
            axis=1
        )

        # ==============================================================
        # STATISTIQUES D'ERREUR
        # ==============================================================

        mae_3d = np.mean(error_norm)

        rmse_3d = np.sqrt(
            np.mean(error_norm ** 2)
        )

        max_error_3d = np.max(error_norm)

        # --------------------------------------------------------------
        # Biais par axe
        # --------------------------------------------------------------

        bias_x = np.mean(error_xyz[:, 0])
        bias_y = np.mean(error_xyz[:, 1])
        bias_z = np.mean(error_xyz[:, 2])

        mae_x = np.mean(
            np.abs(error_xyz[:, 0])
        )

        mae_y = np.mean(
            np.abs(error_xyz[:, 1])
        )

        mae_z = np.mean(
            np.abs(error_xyz[:, 2])
        )

        # ==============================================================
        # DISTANCE AU POINT INITIAL
        # ==============================================================

        dist_gutter = np.linalg.norm(
            gutter_relative,
            axis=1
        )

        dist_diadem = np.linalg.norm(
            diadem_relative,
            axis=1
        )

        # ==============================================================
        # CORRÉLATION
        # ==============================================================

        if (
            np.std(dist_gutter) > EPS
            and np.std(dist_diadem) > EPS
        ):

            try:

                pearson_r, pearson_p = pearsonr(
                    dist_gutter,
                    dist_diadem
                )

            except Exception:

                pearson_r = np.nan
                pearson_p = np.nan

        else:

            pearson_r = np.nan
            pearson_p = np.nan

        # ==============================================================
        # ROM
        # ==============================================================

        rom_gutter_x = np.ptp(
            gutter_relative[:, 0]
        )

        rom_gutter_y = np.ptp(
            gutter_relative[:, 1]
        )

        rom_gutter_z = np.ptp(
            gutter_relative[:, 2]
        )

        rom_diadem_x = np.ptp(
            diadem_relative[:, 0]
        )

        rom_diadem_y = np.ptp(
            diadem_relative[:, 1]
        )

        rom_diadem_z = np.ptp(
            diadem_relative[:, 2]
        )

        rom_gutter_3d = np.ptp(
            dist_gutter
        )

        rom_diadem_3d = np.ptp(
            dist_diadem
        )

        # ==============================================================
        # RÉSULTAT
        # ==============================================================

        marker_results.append({

            "File": file_name,
            "Movement": movement,
            "Marker": marker,
            "N_frames": len(gutter_local),

            "Pearson_R": pearson_r,
            "Pearson_p": pearson_p,

            "MAE_3D_mm": mae_3d,
            "RMSE_3D_mm": rmse_3d,
            "Max_error_3D_mm": max_error_3d,

            "Bias_X_mm": bias_x,
            "Bias_Y_mm": bias_y,
            "Bias_Z_mm": bias_z,

            "MAE_X_mm": mae_x,
            "MAE_Y_mm": mae_y,
            "MAE_Z_mm": mae_z,

            "ROM_G_X_mm": rom_gutter_x,
            "ROM_G_Y_mm": rom_gutter_y,
            "ROM_G_Z_mm": rom_gutter_z,

            "ROM_D_X_mm": rom_diadem_x,
            "ROM_D_Y_mm": rom_diadem_y,
            "ROM_D_Z_mm": rom_diadem_z,

            "ROM_G_3D_mm": rom_gutter_3d,
            "ROM_D_3D_mm": rom_diadem_3d,

            "Error_X_series": error_xyz[:, 0],
            "Error_Y_series": error_xyz[:, 1],
            "Error_Z_series": error_xyz[:, 2],
            "Error_3D_series": error_norm,

            "Dist_G_series": dist_gutter,
            "Dist_D_series": dist_diadem,

            "G_X_series": gutter_relative[:, 0],
            "G_Y_series": gutter_relative[:, 1],
            "G_Z_series": gutter_relative[:, 2],

            "D_X_series": diadem_relative[:, 0],
            "D_Y_series": diadem_relative[:, 1],
            "D_Z_series": diadem_relative[:, 2],
        })

    # =================================================================
    # STABILITÉ DU DIADEME
    # =================================================================

    stability_results = []

    translations = []
    rotations = []
    valid_frames_stability = []

    for frame in range(n_frames):

        gutter_frame = gutter_frames[frame]
        diadem_frame = diadem_frames[frame]

        if (
            gutter_frame is None
            or diadem_frame is None
        ):
            continue

        R_g, origin_g = gutter_frame
        R_d, origin_d = diadem_frame

        translation, rotation = (
            relative_transform_gutter_to_diadem(
                R_g,
                origin_g,
                R_d,
                origin_d
            )
        )

        translations.append(translation)
        rotations.append(rotation)
        valid_frames_stability.append(frame)

    if len(translations) >= MIN_VALID_FRAMES:

        translations = np.asarray(translations)
        rotations = np.asarray(rotations)

        # ==============================================================
        # POSE INITIALE
        # ==============================================================

        translation_initial = translations[0]

        rotation_initial = rotations[0]

        # ==============================================================
        # DRIFT DE TRANSLATION
        # ==============================================================

        translation_drift = (
            translations
            - translation_initial
        )

        translation_drift_norm = np.linalg.norm(
            translation_drift,
            axis=1
        )

        # ==============================================================
        # DRIFT ROTATION
        # ==============================================================

        rotation_drift_deg = []

        for rotation_current in rotations:

            rotation_relative = (
                rotation_initial.T
                @ rotation_current
            )

            angle = rotation_angle_from_matrix(
                rotation_relative
            )

            rotation_drift_deg.append(angle)

        rotation_drift_deg = np.asarray(
            rotation_drift_deg
        )

        # ==============================================================
        # STATISTIQUES
        # ==============================================================

        stability_results.append({

            "File": file_name,
            "Subject": extract_subject(file_name),
            "Movement": movement,
            "N_frames": len(translations),

            # ----------------------------------------------------------
            # Translation
            # ----------------------------------------------------------

            "Trans_X_mean_mm":
                np.mean(translation_drift[:, 0]),

            "Trans_Y_mean_mm":
                np.mean(translation_drift[:, 1]),

            "Trans_Z_mean_mm":
                np.mean(translation_drift[:, 2]),

            "Trans_X_PTP_mm":
                np.ptp(translation_drift[:, 0]),

            "Trans_Y_PTP_mm":
                np.ptp(translation_drift[:, 1]),

            "Trans_Z_PTP_mm":
                np.ptp(translation_drift[:, 2]),

            "Trans_norm_mean_mm":
                np.mean(translation_drift_norm),

            "Trans_norm_std_mm":
                np.std(translation_drift_norm),

            "Trans_norm_max_mm":
                np.max(translation_drift_norm),

            "Trans_norm_PTP_mm":
                np.ptp(translation_drift_norm),

            # ----------------------------------------------------------
            # Rotation
            # ----------------------------------------------------------

            "Rot_drift_mean_deg":
                np.mean(rotation_drift_deg),

            "Rot_drift_std_deg":
                np.std(rotation_drift_deg),

            "Rot_drift_max_deg":
                np.max(rotation_drift_deg),

            "Rot_drift_PTP_deg":
                np.ptp(rotation_drift_deg),

            # ----------------------------------------------------------
            # Séries temporelles
            # ----------------------------------------------------------

            "Trans_X_series":
                translation_drift[:, 0],

            "Trans_Y_series":
                translation_drift[:, 1],

            "Trans_Z_series":
                translation_drift[:, 2],

            "Trans_norm_series":
                translation_drift_norm,

            "Rot_drift_series":
                rotation_drift_deg,
        })

    return marker_results, stability_results


# =====================================================================
# TRAITEMENT GLOBAL
# =====================================================================

all_marker_results = []
all_stability_results = []

gutter_files = sorted(
    DIR_G.glob("*.csv")
)

print("\n")
print("=" * 80)
print("ANALYSE GOUTTIÈRE vs DIADEME")
print("=" * 80)

print(
    f"Fichiers gouttière trouvés : {len(gutter_files)}"
)

for index, gutter_file in enumerate(
    gutter_files,
    start=1
):

    print(
        f"\n[{index}/{len(gutter_files)}] "
        f"{gutter_file.name}"
    )

    # --------------------------------------------------------------
    # Fichier diadème
    # --------------------------------------------------------------

    diadem_file = find_matching_diadem_file(
        gutter_file
    )

    if diadem_file is None:

        print(
            "   ⚠ Aucun fichier diadème correspondant"
        )

        continue

    print(
        f"   ↳ Diadème : {diadem_file.name}"
    )

    # --------------------------------------------------------------
    # Lecture
    # --------------------------------------------------------------

    gutter_data = load_vicon(
        gutter_file
    )

    diadem_data = load_vicon(
        diadem_file
    )

    if (
        gutter_data is None
        or diadem_data is None
    ):

        print(
            "   ⚠ Erreur de lecture"
        )

        continue

    # --------------------------------------------------------------
    # Vérification repères
    # --------------------------------------------------------------

    missing_gutter = [
        marker
        for marker in REF_POINTS
        if marker not in gutter_data
    ]

    missing_diadem = [
        marker
        for marker in REF_POINTS
        if marker not in diadem_data
    ]

    if missing_gutter:

        print(
            f"   ⚠ Repère gouttière incomplet : "
            f"{missing_gutter}"
        )

        continue

    if missing_diadem:

        print(
            f"   ⚠ Repère diadème incomplet : "
            f"{missing_diadem}"
        )

        continue

    # --------------------------------------------------------------
    # Analyse
    # --------------------------------------------------------------

    marker_results, stability_results = (
        analyze_file_pair(
            gutter_data,
            diadem_data,
            gutter_file.name
        )
    )

    all_marker_results.extend(
        marker_results
    )

    all_stability_results.extend(
        stability_results
    )

    print(
        f"   ✓ {len(marker_results)} marqueurs analysés"
    )

    if stability_results:

        print(
            "   ✓ Stabilité du diadème analysée"
        )


# =====================================================================
# DATAFRAMES
# =====================================================================

df_markers = pandas.DataFrame(
    all_marker_results
)

df_stability = pandas.DataFrame(
    all_stability_results
)


# =====================================================================
# VÉRIFICATION
# =====================================================================

print("\n")
print("=" * 80)
print("RÉSULTATS")
print("=" * 80)

print(
    f"Résultats marqueurs : {len(df_markers)}"
)

print(
    f"Résultats stabilité : {len(df_stability)}"
)


# =====================================================================
# EXPORT BRUT
# =====================================================================

if not df_markers.empty:

    scalar_columns = [
        column
        for column in df_markers.columns
        if not isinstance(
            df_markers[column].iloc[0],
            (np.ndarray, list)
        )
    ]

    df_markers[
        scalar_columns
    ].to_csv(
        OUT / "01_Marqueurs_Comparaison.csv",
        index=False
    )


if not df_stability.empty:

    scalar_columns_stab = [
        column
        for column in df_stability.columns
        if not isinstance(
            df_stability[column].iloc[0],
            (np.ndarray, list)
        )
    ]

    df_stability[
        scalar_columns_stab
    ].to_csv(
        OUT / "02_Diademe_Stabilite.csv",
        index=False
    )


# =====================================================================
# TABLEAU SYNTHÈSE PAR MOUVEMENT
# =====================================================================

if not df_markers.empty:

    movement_summary = (
        df_markers
        .groupby("Movement")
        .agg(

            N_measurements=(
                "Marker",
                "count"
            ),

            Pearson_R_mean=(
                "Pearson_R",
                "mean"
            ),

            Pearson_R_median=(
                "Pearson_R",
                "median"
            ),

            MAE_3D_mean_mm=(
                "MAE_3D_mm",
                "mean"
            ),

            MAE_3D_median_mm=(
                "MAE_3D_mm",
                "median"
            ),

            RMSE_3D_mean_mm=(
                "RMSE_3D_mm",
                "mean"
            ),

            RMSE_3D_median_mm=(
                "RMSE_3D_mm",
                "median"
            ),

            Max_error_mean_mm=(
                "Max_error_3D_mm",
                "mean"
            ),

            MAE_X_mean_mm=(
                "MAE_X_mm",
                "mean"
            ),

            MAE_Y_mean_mm=(
                "MAE_Y_mm",
                "mean"
            ),

            MAE_Z_mean_mm=(
                "MAE_Z_mm",
                "mean"
            ),

        )
        .round(4)
    )

    movement_summary.to_csv(
        OUT / "03_Synthese_Comparaison_Mouvements.csv"
    )


# =====================================================================
# TABLEAU SYNTHÈSE PAR MARQUEUR
# =====================================================================

if not df_markers.empty:

    marker_summary = (
        df_markers
        .groupby("Marker")
        .agg(

            N_measurements=(
                "Movement",
                "count"
            ),

            Pearson_R_mean=(
                "Pearson_R",
                "mean"
            ),

            MAE_3D_mean_mm=(
                "MAE_3D_mm",
                "mean"
            ),

            RMSE_3D_mean_mm=(
                "RMSE_3D_mm",
                "mean"
            ),

            Bias_X_mean_mm=(
                "Bias_X_mm",
                "mean"
            ),

            Bias_Y_mean_mm=(
                "Bias_Y_mm",
                "mean"
            ),

            Bias_Z_mean_mm=(
                "Bias_Z_mm",
                "mean"
            ),

        )
        .round(4)
    )

    marker_summary.to_csv(
        OUT / "04_Synthese_Comparaison_Marqueurs.csv"
    )


# =====================================================================
# FIGURE 1
# BLAND-ALTMAN SUR ROM 3D
# =====================================================================

if not df_markers.empty:

    mean_rom = (
        df_markers["ROM_G_3D_mm"]
        + df_markers["ROM_D_3D_mm"]
    ) / 2

    diff_rom = (
        df_markers["ROM_G_3D_mm"]
        - df_markers["ROM_D_3D_mm"]
    )

    bias = np.mean(diff_rom)
    sd = np.std(diff_rom, ddof=1)

    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    plt.figure(figsize=(9, 6))

    plt.scatter(
        mean_rom,
        diff_rom,
        alpha=0.5,
        s=25
    )

    plt.axhline(
        bias,
        linewidth=2,
        label=f"Bias = {bias:.2f} mm"
    )

    plt.axhline(
        loa_upper,
        linestyle="--",
        linewidth=1.5,
        label=f"+1.96 SD = {loa_upper:.2f} mm"
    )

    plt.axhline(
        loa_lower,
        linestyle="--",
        linewidth=1.5,
        label=f"-1.96 SD = {loa_lower:.2f} mm"
    )

    plt.xlabel(
        "Mean ROM 3D [mm]"
    )

    plt.ylabel(
        "Difference Gouttière − Diadème [mm]"
    )

    plt.title(
        "Bland–Altman agreement of 3D ROM"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig01_BlandAltman_ROM3D.png",
        dpi=300
    )

    plt.close()


# =====================================================================
# FIGURE 2
# MAE PAR MOUVEMENT
# =====================================================================

if not df_markers.empty:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df_markers,
        x="Movement",
        y="MAE_3D_mm"
    )

    sns.stripplot(
        data=df_markers,
        x="Movement",
        y="MAE_3D_mm",
        color="black",
        alpha=0.4,
        size=3
    )

    plt.xlabel(
        "Movement"
    )

    plt.ylabel(
        "MAE 3D [mm]"
    )

    plt.title(
        "3D tracking error by movement"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig02_MAE_by_Movement.png",
        dpi=300
    )

    plt.close()


# =====================================================================
# FIGURE 3
# PEARSON PAR MOUVEMENT
# =====================================================================

if not df_markers.empty:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df_markers,
        x="Movement",
        y="Pearson_R"
    )

    sns.stripplot(
        data=df_markers,
        x="Movement",
        y="Pearson_R",
        color="black",
        alpha=0.4,
        size=3
    )

    plt.axhline(
        0,
        linestyle=":"
    )

    plt.xlabel(
        "Movement"
    )

    plt.ylabel(
        "Pearson r"
    )

    plt.title(
        "Temporal correlation of marker displacement"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig03_Pearson_by_Movement.png",
        dpi=300
    )

    plt.close()


# =====================================================================
# FIGURE 4
# ERREUR X/Y/Z
# =====================================================================

if not df_markers.empty:

    axis_data = pandas.DataFrame({

        "X":
            df_markers["MAE_X_mm"],

        "Y":
            df_markers["MAE_Y_mm"],

        "Z":
            df_markers["MAE_Z_mm"],

    })

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=axis_data
    )

    plt.ylabel(
        "MAE [mm]"
    )

    plt.title(
        "Coordinate-wise tracking error"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig04_Error_XYZ.png",
        dpi=300
    )

    plt.close()


# =====================================================================
# FIGURE 5
# EXEMPLE DE TRAJECTOIRE
# =====================================================================

if not df_markers.empty:

    # --------------------------------------------------------------
    # M5 / C02 si disponible
    # --------------------------------------------------------------

    selection = df_markers[
        (df_markers["Movement"] == "M5")
        & (df_markers["Marker"] == "C02")
    ]

    if selection.empty:

        selection = df_markers.iloc[[0]]

    if not selection.empty:

        row = selection.iloc[0]

        plt.figure(figsize=(11, 6))

        plt.plot(
            row["Dist_G_series"],
            linewidth=2,
            label="Gouttière"
        )

        plt.plot(
            row["Dist_D_series"],
            linestyle="--",
            linewidth=2,
            label="Diadème"
        )

        plt.xlabel(
            "Frame"
        )

        plt.ylabel(
            "Displacement from initial position [mm]"
        )

        plt.title(
            f"Trajectory comparison — "
            f"{row['Movement']} — {row['Marker']}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUT / "Fig05_Trajectory_Example.png",
            dpi=300
        )

        plt.close()


# =====================================================================
# STABILITÉ DU DIADEME
# =====================================================================

if not df_stability.empty:

    # =================================================================
    # FIGURE 6 — DRIFT TRANSLATION 3D
    # =================================================================

    plt.figure(figsize=(10, 6))

    for _, row in df_stability.iterrows():

        plt.plot(
            row["Trans_norm_series"],
            alpha=0.5
        )

    plt.axhline(
        TRANSLATION_TOLERANCE_MM,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f'Tolerance = {TRANSLATION_TOLERANCE_MM} mm'
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Translation drift [mm]"
    )

    plt.title(
        "Diadem translation drift in gutter frame"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig06_Diadem_Translation_Drift.png",
        dpi=300
    )

    plt.close()

    # =================================================================
    # FIGURE 7 — DRIFT ROTATION
    # =================================================================

    plt.figure(figsize=(10, 6))

    for _, row in df_stability.iterrows():

        plt.plot(
            row["Rot_drift_series"],
            alpha=0.5
        )

    plt.axhline(
        ROTATION_TOLERANCE_DEG,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f'Tolerance = {ROTATION_TOLERANCE_DEG}°'
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Rotation drift [deg]"
    )

    plt.title(
        "Diadem rotational drift in gutter frame"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig07_Diadem_Rotation_Drift.png",
        dpi=300
    )

    plt.close()

    # =================================================================
    # FIGURE 8 — TRANSLATION X/Y/Z
    # =================================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True
    )

    for _, row in df_stability.iterrows():

        axes[0].plot(
            row["Trans_X_series"],
            alpha=0.4
        )

        axes[1].plot(
            row["Trans_Y_series"],
            alpha=0.4
        )

        axes[2].plot(
            row["Trans_Z_series"],
            alpha=0.4
        )

    axes[0].axhline(
        0,
        linestyle=":"
    )

    axes[1].axhline(
        0,
        linestyle=":"
    )

    axes[2].axhline(
        0,
        linestyle=":"
    )

    axes[0].set_ylabel(
        "ΔX [mm]"
    )

    axes[1].set_ylabel(
        "ΔY [mm]"
    )

    axes[2].set_ylabel(
        "ΔZ [mm]"
    )

    axes[2].set_xlabel(
        "Frame"
    )

    fig.suptitle(
        "Diadem translation drift by axis"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig08_Diadem_Drift_XYZ.png",
        dpi=300
    )

    plt.close()

    # =================================================================
    # FIGURE 9 — BOXPLOT TRANSLATION
    # =================================================================

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df_stability,
        x="Movement",
        y="Trans_norm_PTP_mm"
    )

    sns.stripplot(
        data=df_stability,
        x="Movement",
        y="Trans_norm_PTP_mm",
        color="black",
        alpha=0.5,
        size=4
    )

    plt.axhline(
        TRANSLATION_TOLERANCE_MM,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f'Tolerance = {TRANSLATION_TOLERANCE_MM} mm'
    )

    plt.xlabel(
        "Movement"
    )

    plt.ylabel(
        "Translation drift PTP [mm]"
    )

    plt.title(
        "Diadem translation drift by movement"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig09_Translation_PTP_by_Movement.png",
        dpi=300
    )

    plt.close()

    # =================================================================
    # FIGURE 10 — BOXPLOT ROTATION
    # =================================================================

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df_stability,
        x="Movement",
        y="Rot_drift_PTP_deg"
    )

    sns.stripplot(
        data=df_stability,
        x="Movement",
        y="Rot_drift_PTP_deg",
        color="black",
        alpha=0.5,
        size=4
    )

    plt.axhline(
        ROTATION_TOLERANCE_DEG,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f'Tolerance = {ROTATION_TOLERANCE_DEG}°'
    )

    plt.xlabel(
        "Movement"
    )

    plt.ylabel(
        "Rotation drift PTP [deg]"
    )

    plt.title(
        "Diadem rotational drift by movement"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig10_Rotation_PTP_by_Movement.png",
        dpi=300
    )

    plt.close()

    # =================================================================
    # FIGURE 11 — TRAJECTOIRE 3D
    # =================================================================

    figure = plt.figure(
        figsize=(10, 8)
    )

    axis = figure.add_subplot(
        111,
        projection="3d"
    )

    for _, row in df_stability.iterrows():

        xs = row["Trans_X_series"]
        ys = row["Trans_Y_series"]
        zs = row["Trans_Z_series"]

        axis.plot(
            xs,
            ys,
            zs,
            alpha=0.5,
            linewidth=1
        )

        # Marqueur de départ (vert) et d'arrivée (rouge)
        axis.scatter(xs[0], ys[0], zs[0], color='green', s=12)
        axis.scatter(xs[-1], ys[-1], zs[-1], color='red', s=12)

    axis.set_xlabel(
        "ΔX [mm]"
    )

    axis.set_ylabel(
        "ΔY [mm]"
    )

    axis.set_zlabel(
        "ΔZ [mm]"
    )

    axis.set_title(
        "3D trajectory of diadem drift (green = start, red = end)"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "Fig11_Diadem_Drift_3D.png",
        dpi=300
    )

    plt.close()

    # =================================================================
    # FIGURE 12 — VARIABILITÉ INTER-SUJETS
    # =================================================================

    if "Subject" in df_stability.columns:

        # Trie les sujets par PTP médian
        subject_summary = (
            df_stability
            .groupby("Subject")["Trans_norm_PTP_mm"]
            .agg(["mean", "median", "count"])
            .sort_values("median")
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Boxplot par sujet — Translation
        sns.boxplot(
            data=df_stability,
            x="Subject",
            y="Trans_norm_PTP_mm",
            order=subject_summary.index,
            palette="Blues",
            ax=axes[0]
        )

        sns.stripplot(
            data=df_stability,
            x="Subject",
            y="Trans_norm_PTP_mm",
            order=subject_summary.index,
            color="black",
            alpha=0.6,
            size=4,
            ax=axes[0]
        )

        axes[0].axhline(
            TRANSLATION_TOLERANCE_MM,
            color='red',
            linestyle='--',
            linewidth=1.2,
            label=f'Tolerance = {TRANSLATION_TOLERANCE_MM} mm'
        )

        axes[0].set_xlabel("Subject")
        axes[0].set_ylabel("Translation drift PTP [mm]")
        axes[0].set_title("Translation drift per subject")
        axes[0].legend()

        # Boxplot par sujet — Rotation
        sns.boxplot(
            data=df_stability,
            x="Subject",
            y="Rot_drift_PTP_deg",
            order=subject_summary.index,
            palette="Oranges",
            ax=axes[1]
        )

        sns.stripplot(
            data=df_stability,
            x="Subject",
            y="Rot_drift_PTP_deg",
            order=subject_summary.index,
            color="black",
            alpha=0.6,
            size=4,
            ax=axes[1]
        )

        axes[1].axhline(
            ROTATION_TOLERANCE_DEG,
            color='red',
            linestyle='--',
            linewidth=1.2,
            label=f'Tolerance = {ROTATION_TOLERANCE_DEG}°'
        )

        axes[1].set_xlabel("Subject")
        axes[1].set_ylabel("Rotation drift PTP [deg]")
        axes[1].set_title("Rotational drift per subject")
        axes[1].legend()

        plt.suptitle(
            "Inter-subject variability of diadem stability",
            fontsize=13,
            fontweight='bold'
        )

        plt.tight_layout()

        plt.savefig(
            OUT / "Fig12_Stability_Per_Subject.png",
            dpi=300
        )

        plt.close()


# =====================================================================
# TABLEAU STABILITÉ PAR MOUVEMENT
# =====================================================================

if not df_stability.empty:

    stability_summary = (
        df_stability
        .groupby("Movement")
        .agg(

            N_files=(
                "File",
                "count"
            ),

            Translation_mean_mm=(
                "Trans_norm_mean_mm",
                "mean"
            ),

            Translation_median_mm=(
                "Trans_norm_mean_mm",
                "median"
            ),

            Translation_PTP_mean_mm=(
                "Trans_norm_PTP_mm",
                "mean"
            ),

            Translation_PTP_median_mm=(
                "Trans_norm_PTP_mm",
                "median"
            ),

            Translation_PTP_SD_mm=(
                "Trans_norm_PTP_mm",
                "std"
            ),

            Rotation_mean_deg=(
                "Rot_drift_mean_deg",
                "mean"
            ),

            Rotation_median_deg=(
                "Rot_drift_mean_deg",
                "median"
            ),

            Rotation_PTP_mean_deg=(
                "Rot_drift_PTP_deg",
                "mean"
            ),

            Rotation_PTP_median_deg=(
                "Rot_drift_PTP_deg",
                "median"
            ),

            Rotation_PTP_SD_deg=(
                "Rot_drift_PTP_deg",
                "std"
            ),

        )
        .round(3)
    )

    stability_summary.to_csv(
        OUT / "05_Stabilite_Diademe_Mouvements.csv"
    )


# =====================================================================
# TABLEAU PUBLICATION
# =====================================================================

if not df_stability.empty:

    publication_rows = []

    for movement, group in df_stability.groupby(
        "Movement"
    ):

        row = {

            "Movement":
                movement,

            "N_files":
                len(group),

            "Translation_mean_SD_mm":
                (
                    f"{group['Trans_norm_mean_mm'].mean():.2f} "
                    f"± "
                    f"{group['Trans_norm_mean_mm'].std():.2f}"
                ),

            "Translation_PTP_mean_SD_mm":
                (
                    f"{group['Trans_norm_PTP_mm'].mean():.2f} "
                    f"± "
                    f"{group['Trans_norm_PTP_mm'].std():.2f}"
                ),

            "Translation_PTP_median_IQR_mm":
                (
                    f"{group['Trans_norm_PTP_mm'].median():.2f} "
                    f"["
                    f"{group['Trans_norm_PTP_mm'].quantile(0.25):.2f}"
                    f"–"
                    f"{group['Trans_norm_PTP_mm'].quantile(0.75):.2f}"
                    f"]"
                ),

            "Rotation_mean_SD_deg":
                (
                    f"{group['Rot_drift_mean_deg'].mean():.2f} "
                    f"± "
                    f"{group['Rot_drift_mean_deg'].std():.2f}"
                ),

            "Rotation_PTP_mean_SD_deg":
                (
                    f"{group['Rot_drift_PTP_deg'].mean():.2f} "
                    f"± "
                    f"{group['Rot_drift_PTP_deg'].std():.2f}"
                ),

            "Rotation_PTP_median_IQR_deg":
                (
                    f"{group['Rot_drift_PTP_deg'].median():.2f} "
                    f"["
                    f"{group['Rot_drift_PTP_deg'].quantile(0.25):.2f}"
                    f"–"
                    f"{group['Rot_drift_PTP_deg'].quantile(0.75):.2f}"
                    f"]"
                ),
        }

        publication_rows.append(row)

    publication_table = pandas.DataFrame(
        publication_rows
    )

    publication_table.to_csv(
        OUT / "06_Stabilite_Diademe_Publication.csv",
        index=False
    )


# =====================================================================
# TABLEAU PAR SUJET (variabilité inter-sujets)
# =====================================================================

if not df_stability.empty and "Subject" in df_stability.columns:

    subject_summary_table = (
        df_stability
        .groupby("Subject")
        .agg(

            N_sessions=("File", "count"),

            Translation_mean_mm=("Trans_norm_mean_mm", "mean"),
            Translation_SD_mm=("Trans_norm_mean_mm", "std"),
            Translation_PTP_mean_mm=("Trans_norm_PTP_mm", "mean"),
            Translation_PTP_median_mm=("Trans_norm_PTP_mm", "median"),
            Translation_PTP_max_mm=("Trans_norm_PTP_mm", "max"),

            Rotation_mean_deg=("Rot_drift_mean_deg", "mean"),
            Rotation_SD_deg=("Rot_drift_mean_deg", "std"),
            Rotation_PTP_mean_deg=("Rot_drift_PTP_deg", "mean"),
            Rotation_PTP_median_deg=("Rot_drift_PTP_deg", "median"),
            Rotation_PTP_max_deg=("Rot_drift_PTP_deg", "max"),

        )
        .round(3)
        .sort_values("Translation_PTP_median_mm")
    )

    subject_summary_table.to_csv(
        OUT / "09_Stabilite_Diademe_Par_Sujet.csv"
    )


# =====================================================================
# STATISTIQUES ENTRE MOUVEMENTS
# =====================================================================

if not df_stability.empty:

    movements = sorted(
        df_stability["Movement"].dropna().unique()
    )

    if len(movements) >= 2:

        translation_groups = [
            group["Trans_norm_PTP_mm"].dropna().values
            for _, group in df_stability.groupby(
                "Movement"
            )
            if len(group["Trans_norm_PTP_mm"].dropna()) > 0
        ]

        rotation_groups = [
            group["Rot_drift_PTP_deg"].dropna().values
            for _, group in df_stability.groupby(
                "Movement"
            )
            if len(group["Rot_drift_PTP_deg"].dropna()) > 0
        ]

        with open(
            OUT / "07_Statistiques_Mouvements.txt",
            "w",
            encoding="utf-8"
        ) as stats_file:

            stats_file.write(
                "STATISTIQUES STABILITE DU DIADEME\n"
            )

            stats_file.write(
                "=" * 70 + "\n\n"
            )

            stats_file.write(
                "1. TEST DE KRUSKAL-WALLIS (non apparié)\n"
            )

            stats_file.write(
                "-" * 70 + "\n"
            )

            stats_file.write(
                "Hypothèse : observations indépendantes.\n"
                "Ce test est fourni à titre indicatif. Comme les mêmes sujets\n"
                "réalisent plusieurs mouvements, les données sont appariées\n"
                "et le test de Friedman (ci-dessous) est préféré.\n\n"
            )

            # ----------------------------------------------------------
            # Translation
            # ----------------------------------------------------------

            if len(translation_groups) >= 2:

                try:

                    H_translation, p_translation = kruskal(
                        *translation_groups
                    )

                    stats_file.write(
                        "Translation drift PTP\n"
                    )

                    stats_file.write(
                        f"H = {H_translation:.5f}\n"
                    )

                    stats_file.write(
                        f"p = {p_translation:.6f}\n\n"
                    )

                except Exception as error:

                    stats_file.write(
                        f"Erreur translation : {error}\n\n"
                    )

            # ----------------------------------------------------------
            # Rotation
            # ----------------------------------------------------------

            if len(rotation_groups) >= 2:

                try:

                    H_rotation, p_rotation = kruskal(
                        *rotation_groups
                    )

                    stats_file.write(
                        "Rotation drift PTP\n"
                    )

                    stats_file.write(
                        f"H = {H_rotation:.5f}\n"
                    )

                    stats_file.write(
                        f"p = {p_rotation:.6f}\n\n"
                    )

                except Exception as error:

                    stats_file.write(
                        f"Erreur rotation : {error}\n\n"
                    )

            # ----------------------------------------------------------
            # FRIEDMAN (apparié par sujet)
            # ----------------------------------------------------------

            stats_file.write(
                "\n2. TEST DE FRIEDMAN (apparié par sujet)\n"
            )

            stats_file.write(
                "-" * 70 + "\n"
            )

            stats_file.write(
                "Hypothèse : les mêmes sujets réalisent tous les mouvements.\n"
                "Seuls les sujets avec données complètes sont inclus.\n\n"
            )

            try:

                pivot_trans = df_stability.pivot_table(
                    index="Subject",
                    columns="Movement",
                    values="Trans_norm_PTP_mm"
                ).dropna()

                pivot_rot = df_stability.pivot_table(
                    index="Subject",
                    columns="Movement",
                    values="Rot_drift_PTP_deg"
                ).dropna()

                stats_file.write(
                    f"N sujets complets (translation) : {len(pivot_trans)}\n"
                )

                stats_file.write(
                    f"N sujets complets (rotation)    : {len(pivot_rot)}\n\n"
                )

                if len(pivot_trans) >= 2 and pivot_trans.shape[1] >= 3:

                    stat_t, p_t = friedmanchisquare(
                        *[pivot_trans[c].values for c in pivot_trans.columns]
                    )

                    stats_file.write(
                        "Translation drift PTP (Friedman)\n"
                    )

                    stats_file.write(
                        f"chi2 = {stat_t:.5f}\n"
                    )

                    stats_file.write(
                        f"p    = {p_t:.6f}\n\n"
                    )

                else:

                    stats_file.write(
                        "Translation : nombre de sujets ou de conditions insuffisant.\n\n"
                    )

                if len(pivot_rot) >= 2 and pivot_rot.shape[1] >= 3:

                    stat_r, p_r = friedmanchisquare(
                        *[pivot_rot[c].values for c in pivot_rot.columns]
                    )

                    stats_file.write(
                        "Rotation drift PTP (Friedman)\n"
                    )

                    stats_file.write(
                        f"chi2 = {stat_r:.5f}\n"
                    )

                    stats_file.write(
                        f"p    = {p_r:.6f}\n\n"
                    )

                else:

                    stats_file.write(
                        "Rotation : nombre de sujets ou de conditions insuffisant.\n\n"
                    )

            except Exception as error:

                stats_file.write(
                    f"Erreur Friedman : {error}\n"
                )


# =====================================================================
# TABLEAU GLOBAL
# =====================================================================

if not df_stability.empty:

    global_summary = pandas.DataFrame([{

        "N_files":
            len(df_stability),

        "Translation_mean_mm":
            df_stability[
                "Trans_norm_mean_mm"
            ].mean(),

        "Translation_SD_mm":
            df_stability[
                "Trans_norm_mean_mm"
            ].std(),

        "Translation_PTP_mean_mm":
            df_stability[
                "Trans_norm_PTP_mm"
            ].mean(),

        "Translation_PTP_median_mm":
            df_stability[
                "Trans_norm_PTP_mm"
            ].median(),

        "Rotation_mean_deg":
            df_stability[
                "Rot_drift_mean_deg"
            ].mean(),

        "Rotation_SD_deg":
            df_stability[
                "Rot_drift_mean_deg"
            ].std(),

        "Rotation_PTP_mean_deg":
            df_stability[
                "Rot_drift_PTP_deg"
            ].mean(),

        "Rotation_PTP_median_deg":
            df_stability[
                "Rot_drift_PTP_deg"
            ].median(),

    }])

    global_summary.to_csv(
        OUT / "08_Stabilite_Diademe_Global.csv",
        index=False
    )


# =====================================================================
# RAPPORT CONSOLE
# =====================================================================

print("\n")
print("=" * 80)
print("ANALYSE TERMINÉE")
print("=" * 80)

print(
    f"\nRésultats enregistrés dans :\n{OUT}"
)

print(
    f"\nNombre de mesures marqueurs : "
    f"{len(df_markers)}"
)

print(
    f"Nombre de sessions stabilité : "
    f"{len(df_stability)}"
)


# =====================================================================
# RÉSUMÉ STABILITÉ
# =====================================================================

if not df_stability.empty:

    print("\n")
    print("=" * 80)
    print("STABILITÉ DU DIADEME")
    print("=" * 80)

    print(
        "\nTranslation drift 3D :"
    )

    print(
        f"   Moyenne : "
        f"{df_stability['Trans_norm_mean_mm'].mean():.3f} mm"
    )

    print(
        f"   Médiane : "
        f"{df_stability['Trans_norm_mean_mm'].median():.3f} mm"
    )

    print(
        f"   PTP médiane : "
        f"{df_stability['Trans_norm_PTP_mm'].median():.3f} mm"
    )

    print(
        f"   PTP max    : "
        f"{df_stability['Trans_norm_PTP_mm'].max():.3f} mm"
    )

    print(
        "\nRotation drift :"
    )

    print(
        f"   Moyenne : "
        f"{df_stability['Rot_drift_mean_deg'].mean():.3f}°"
    )

    print(
        f"   Médiane : "
        f"{df_stability['Rot_drift_mean_deg'].median():.3f}°"
    )

    print(
        f"   PTP médiane : "
        f"{df_stability['Rot_drift_PTP_deg'].median():.3f}°"
    )

    print(
        f"   PTP max    : "
        f"{df_stability['Rot_drift_PTP_deg'].max():.3f}°"
    )

    # --------------------------------------------------------------
    # Bilan tolérance
    # --------------------------------------------------------------

    n_files = len(df_stability)
    n_trans_ok = (df_stability["Trans_norm_PTP_mm"] <= TRANSLATION_TOLERANCE_MM).sum()
    n_rot_ok = (df_stability["Rot_drift_PTP_deg"] <= ROTATION_TOLERANCE_DEG).sum()

    print("\nBilan tolérance :")
    print(
        f"   Translation ≤ {TRANSLATION_TOLERANCE_MM} mm : "
        f"{n_trans_ok}/{n_files} sessions "
        f"({100 * n_trans_ok / n_files:.1f}%)"
    )
    print(
        f"   Rotation    ≤ {ROTATION_TOLERANCE_DEG}° : "
        f"{n_rot_ok}/{n_files} sessions "
        f"({100 * n_rot_ok / n_files:.1f}%)"
    )


# =====================================================================
# FIN
# =====================================================================

print("\n")
print("=" * 80)
print("FICHIERS PRINCIPAUX")
print("=" * 80)

print("01_Marqueurs_Comparaison.csv")
print("02_Diademe_Stabilite.csv")
print("03_Synthese_Comparaison_Mouvements.csv")
print("04_Synthese_Comparaison_Marqueurs.csv")
print("05_Stabilite_Diademe_Mouvements.csv")
print("06_Stabilite_Diademe_Publication.csv")
print("07_Statistiques_Mouvements.txt")
print("08_Stabilite_Diademe_Global.csv")
print("09_Stabilite_Diademe_Par_Sujet.csv")

print("\n")
print("✓ Analyse terminée.")
print("=" * 80)