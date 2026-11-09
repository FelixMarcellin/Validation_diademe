# Validation of a 3D-printed headband as a cranial reference for facial motion capture

## 1. Overview

This repository contains the Python pipeline used to compare two cranial reference systems for marker-based facial motion capture:

- **Dental splint (G):** a maxillary dental splint equipped with three retroreflective markers.
- **Headband (D):** a 3D-printed headband equipped with three retroreflective markers.

The objective is to investigate whether the 3D-printed headband can provide a reliable alternative to the maxillary dental splint for the analysis of facial kinematics.

The comparison focuses on:

1. Geometric configuration and rigidity of each reference system.
2. Relative translation and rotation between the two reference systems during facial movements.
3. Agreement between facial kinematic measurements expressed in each reference frame.
4. Residual kinematic variability in markers classified as non-moving.
5. Sensitivity of the results to filtering parameters.
6. A complementary skin-based reference analysis.

The pipeline processes synchronized Vicon recordings and generates quantitative results, CSV tables and figures for further statistical analysis and scientific reporting.

**Current pipeline version: V11**

## 2. Repository contents

```text
Validation_diademe/
│
├── README.md
│
└── comparaison_repere_V11_reperes faciaux_simplifiee.py
```

The experimental data and generated results are not included in this repository.

Recommended local working structure:

```text
Validation_diademe/
│
├── comparaison_repere_V11_reperes faciaux_simplifiee.py
├── README.md
│
├── DATA/
│   ├── gouttiere/
│   │   └── paired Vicon CSV files
│   │
│   └── diademe/
│       └── paired Vicon CSV files
│
└── RESULTATS/
    └── RESULTATS_COMPARAISON_V11_DATADRIVEN/
        ├── CSV tables
        └── PNG figures
```

## 3. Experimental dataset

The study uses simultaneous facial motion capture recordings obtained with both reference systems.

The two systems are recorded by the same Vicon motion capture system. Consequently, corresponding recordings are synchronized frame by frame and do not require temporal realignment.

The dataset used in the study comprises 9 participants and 49 paired trials.

### 3.1. Reference markers

Each reference system is defined by three retroreflective markers:

| Reference system | Markers | Diameter |
|---|---|---:|
| Dental splint | M1, M2, M3 | 5 mm |
| 3D-printed headband | M1, M2, M3 | 14 mm |

The facial markers have a nominal diameter of 1.5 mm.

Marker names can be adapted in the configuration section of the Python script through `REF_POINTS_G` and `REF_POINTS_D`.

### 3.2. Facial movements

The current analysis focuses on five standardized facial movements:

| Code | Movement |
|---|---|
| M1 | Simple eye closure |
| M2 | Forced eye closure |
| M3 | Lip protrusion ("o") |
| M4 | Lip protrusion ("pou") |
| M5 | Broad smile |

The analysis uses the facial markers shared by the two recordings.

### 3.3. File naming and pairing

The pipeline automatically searches for the corresponding headband recording for each dental splint recording.

The matching procedure relies on the replacement of the device identifier in the filename:

- `_G_` is replaced by `_D_`
- `_g_` is replaced by `_d_`
- Additional replacement patterns are also implemented in the script.

Example:

```text
20260824_G_SMA_M1.csv
20260824_D_SMA_M1.csv
```

The two files must correspond to the same participant, movement and acquisition.

The subject and movement identifiers are extracted from the filename. The naming convention must therefore remain consistent across the dataset.

### 3.4. Vicon CSV format

The script reads Vicon CSV files using the following structure:

- The first four rows are skipped when loading numerical data.
- The second line contains the acquisition frequency.
- The third line contains marker names, using the Vicon `Labelisation_Standard:NAME` convention.
- Numerical coordinates are read as three columns per marker (X, Y, Z).

The acquisition frequency is read from the file when available. A default value of 100 Hz is used if it cannot be retrieved.

## 4. Software requirements

The pipeline uses Python 3.9 or later and the following libraries:

- NumPy
- pandas
- SciPy
- Matplotlib
- Seaborn

Install the dependencies with:

```bash
pip install numpy pandas scipy matplotlib seaborn
```

The script also uses standard Python modules, including `pathlib`.

For reproducibility, the Python version and package versions used for each analysis should be recorded. The repository does not currently provide a pinned dependency file.

## 5. Configuration

All analysis parameters are defined in the `CONFIGURATION` section of the Python script.

The main parameters are summarized below.

| Parameter | V11 value | Description |
|---|---:|---|
| `REF_POINTS_G` | M1, M2, M3 | Dental splint reference markers |
| `REF_POINTS_D` | M1, M2, M3 | Headband reference markers |
| `FPS_DEFAULT` | 100 Hz | Default acquisition frequency |
| `REPERE_MODE` | `kabsch` | Reference-frame construction |
| `BASELINE_FRAMES` | 5 | Frames used for baseline correction |
| `FILTER_CUTOFF_HZ` | 6 Hz | Default low-pass cutoff |
| `FILTER_ORDER` | 2 | Butterworth filter order |
| `EDGE_TRIM_FRAMES` | 5 | Frames removed at each signal edge |
| `SWEEP_CUTOFFS_HZ` | 2, 3, 4, 5, 6, 8, 10 Hz | Cutoff sensitivity analysis |
| `MAX_GAP_FRAMES` | 10 | Maximum gap length interpolated |
| `MIN_RUN_PCT` | 70% | Minimum longest valid run |
| `MAX_INTERP_PCT` | 15% | Maximum interpolated-frame proportion |
| `MIN_VALID_FRAMES` | 30 | Minimum valid frames |
| `AMP_MOVING_FACTOR` | 5 | Noise multiplier for movement classification |
| `AMP_MOVING_FLOOR_MM` | 1 mm | Minimum movement threshold |
| `FIXED_THRESHOLD_MM` | 1.5 mm | Threshold for skin-reference marker selection |
| `MIN_FIXED_MARKERS` | 10 | Minimum markers for skin reference |
| `MIN_SKIN_FRAMES` | 30 | Minimum valid frames for skin amplitude |
| `EQUIVALENCE_DELTA_MM` | 2 mm | TOST equivalence margin for displacement range |
| `ALIGN_ORIENTATION` | `True` | Alignment of headband and splint orientations |

### 5.1. Local paths

The input and output directories are defined by:

```python
DIR_G = Path(r"...\DATA\gouttiere")
DIR_D = Path(r"...\DATA\diademe")
OUT = Path(r"...\RESULTATS\RESULTATS_COMPARAISON_V11_DATADRIVEN")
```

These paths must be adapted to the local computer before running the analysis.

## 6. Processing and analysis methods

### 6.1. Reference-frame construction

The three reference markers M1, M2 and M3 define the local coordinate system for each device.

The recommended configuration is:

```python
REPERE_MODE = "kabsch"
```

The Kabsch algorithm estimates the rigid transformation that best fits the reference triangle to its observed position at each frame.

This approach uses the three reference markers jointly to estimate translation and rotation, rather than relying on a single marker as the origin.

The local coordinate systems are used to express facial marker trajectories relative to the corresponding reference device.

When enabled, `ALIGN_ORIENTATION = True` aligns the mean headband orientation with the dental splint orientation before comparing the kinematic vectors.

### 6.2. Baseline correction

Facial displacement is calculated relative to the mean position over the first five valid frames.

This reduces the influence of single-frame positional variability when defining the initial position.

The baseline correction is applied to displacement signals. Velocity and acceleration are calculated from the filtered trajectories.

### 6.3. Filtering and kinematic calculations

A zero-phase Butterworth low-pass filter is applied to the coordinate trajectories.

The default configuration is:

- Cutoff frequency: 6 Hz.
- Filter order: 2.
- Zero-phase filtering using `filtfilt`.

Velocity and acceleration are calculated using numerical gradients:

\[
v(t)=\frac{dP(t)}{dt}
\]

\[
a(t)=\frac{dv(t)}{dt}
\]

Five frames are removed at each edge after filtering and differentiation to limit edge-related effects.

The analysis is performed for:

- Displacement (mm).
- Velocity (mm/s).
- Acceleration (mm/s²).

### 6.4. Data-driven classification of facial markers

Unlike approaches based on a predefined anatomical list of moving markers, V11 classifies markers separately for each trial.

For each facial marker, the amplitude is calculated in the local coordinate system of each device using the median of the coordinate-wise peak-to-peak ranges.

The movement threshold is defined as:

\[
T_{\mathrm{moving}} =
\max(5 \times \sigma_{\mathrm{face}}, 1\ \mathrm{mm})
\]

where \(\sigma_{\mathrm{face}}\) is the estimated facial marker variability used by the pipeline.

A marker is classified as moving when its amplitude exceeds this threshold in **both** reference systems.

The intersection of the two classifications is used to reduce the risk of identifying movement that appears in only one reference system.

The remaining markers are classified as non-moving by the current implementation. This category should be interpreted cautiously: markers with insufficient valid data may also fall outside the moving classification and should not automatically be considered truly stationary.

The classification and amplitude values are exported for inspection.

### 6.5. Reference-system rigidity

The geometric and temporal stability of each reference triangle are assessed using:

- Inter-marker distances (M1–M2, M1–M3 and M2–M3).
- Triangle height.
- Triangle area.
- Kabsch fitting RMSD.

The inter-marker distances, triangle height and area characterize the geometry of the reference system.

The Kabsch RMSD quantifies the residual discrepancy between the reference triangle and its observed configuration over time.

These measurements provide complementary information about geometric configuration and temporal rigidity.

### 6.6. Relative stability of the two reference systems

The relative transformation between the dental splint and headband is calculated throughout each recording.

The analysis quantifies:

- Relative translation.
- Translation drift over time.
- Relative rotation.
- Rotation drift over time.
- Translation drift speed.

Translation stability is evaluated using 2 mm and 3 mm descriptive thresholds.

Rotation stability is evaluated using a 2° threshold.

These thresholds are analysis parameters and should not be interpreted as universal clinical acceptance limits.

### 6.7. Skin-based reference analysis

A complementary analysis uses facial markers with low-amplitude trajectories to construct a skin-based reference.

A marker is selected when its amplitude remains below 1.5 mm in both the dental splint and headband reference systems.

At least 10 selected markers are required to construct the skin reference.

The resulting reference is used to estimate the relative motion of the dental splint and headband with respect to the facial skin.

This analysis provides an additional perspective on relative device movement. It is not an independent gold standard: the skin is deformable, and marker selection is based on amplitudes measured in the two device reference frames.

The results must therefore be interpreted as a complementary consistency analysis rather than an absolute validation against an independent reference.

### 6.8. Vectorial comparison of facial kinematics

For each common facial marker and movement, the pipeline compares the kinematic vectors obtained from the two reference systems.

The comparison includes:

- Mean absolute vector error (MAE).
- Root mean square vector error (RMSE).
- Maximum vector error.
- Axis-specific bias and RMSE.
- Difference between vector magnitudes.
- Peak magnitude difference.
- Relative peak error.
- Normalized RMSE.
- Pearson correlation between vector magnitudes.
- Mean, median and maximum angular difference between vectors.

The angular comparison is performed only when both vector magnitudes exceed the minimum thresholds defined for the corresponding kinematic variable.

The reported range of motion (ROM) is calculated as the peak-to-peak range of the displacement magnitude over time:

\[
ROM = \max(\|P(t)\|)-\min(\|P(t)\|)
\]

It should therefore be interpreted as the range of the displacement magnitude, rather than as a general three-dimensional anatomical range of motion.

### 6.9. Filter cut-off sensitivity analysis

The pipeline repeats the kinematic calculations using several low-pass filter cut-off frequencies:

```python
SWEEP_CUTOFFS_HZ = [2, 3, 4, 5, 6, 8, 10]
```

The purpose is to evaluate how the estimated kinematic quantities vary with filtering choices.

This analysis is particularly relevant for velocity and acceleration, which are more sensitive to high-frequency noise than displacement.

The sensitivity analysis should be interpreted as an assessment of parameter dependence. Stabilization of a peak value alone does not establish that the underlying biological signal has been preserved.

### 6.10. Agreement and statistical analyses

The pipeline includes agreement analyses based on the differences between measurements obtained with the two reference systems.

The Bland–Altman analysis evaluates the bias and limits of agreement for the selected kinematic outcome.

The TOST procedure is configured with an equivalence margin of ±2 mm for the displacement-range outcome.

The equivalence margin is a methodological choice and requires an independent clinical or biomechanical justification before being used to support a claim of interchangeability.

The script also includes paired and movement-related statistical comparisons. The statistical unit must be considered carefully: measurements from multiple markers and trials belonging to the same participant are not independent observations.

For this reason, pooled marker-level results should primarily be interpreted as descriptive unless the statistical model explicitly accounts for the repeated-measures structure.

## 7. Running the pipeline

After configuring the input and output paths, run:

```bash
python "comparaison_repere_V11_reperes faciaux_simplifiee.py"
```

The pipeline searches for dental splint recordings, identifies the corresponding headband recordings, loads the paired data and performs the analysis.

The output directory is created or populated at the location specified by `OUT`.

The generated CSV files and figures should be checked after execution, particularly for:

- Missing or unmatched recordings.
- Missing reference markers.
- Insufficient valid frames.
- Marker coverage and exclusions.
- Quality-control flags.
- Number of markers classified as moving or non-moving.
- Availability of a sufficient number of markers for the skin reference.

## 8. Outputs

The pipeline generates tables and figures covering the different stages of the analysis.

The principal output categories are:

| Output category | Information |
|---|---|
| Marker-level comparison | Agreement between splint and headband kinematics for each facial marker and trial |
| Movement-level summaries | Aggregated kinematic agreement for each facial movement |
| Global kinematic summaries | Displacement, velocity and acceleration comparison |
| Reference stability | Relative translation and rotation between devices |
| Subject and movement stability | Distribution of reference drift across participants and movements |
| Marker classification | Moving/non-moving classification and corresponding amplitudes |
| Marker coverage | Available markers and exclusions |
| Static-marker variability | Residual displacement, velocity and acceleration variability |
| Filter sensitivity | Effect of filter cut-off frequency on kinematic measurements |
| Reference rigidity | Triangle geometry and Kabsch RMSD |
| Skin-reference analysis | Selection of low-amplitude facial markers and relative device drift |
| Bland–Altman analysis | Bias and limits of agreement for selected outcomes |

Figures are generated as PNG files and tables are exported as CSV files. The exact filenames are defined in the current script and should be treated as the authoritative list of outputs.

## 9. Interpretation and limitations

Several methodological considerations are important when interpreting the results.

### 9.1. Sample size and repeated measurements

The study includes a limited number of participants, with multiple trials and facial markers per participant.

The observations are therefore hierarchically structured. A large number of marker-level measurements does not compensate for a limited number of independent participants.

Statistical conclusions should account for this structure, and inferential results should be interpreted cautiously.

### 9.2. Reference-system independence

The dental splint is used as the comparison reference, but it should not be assumed to be perfectly stable.

Small movements of the splint relative to the maxilla may occur during facial contractions.

The study therefore evaluates relative agreement and stability rather than assuming that the dental splint represents an error-free gold standard.

### 9.3. Skin compliance

The skin-based reference is affected by soft-tissue deformation and the availability of sufficiently stable facial markers.

It provides complementary information but cannot be considered an independent absolute reference.

### 9.4. Marker classification

The data-driven classification depends on the estimated marker variability and the selected amplitude thresholds.

Markers classified as non-moving should not automatically be interpreted as physically stationary, particularly when data are missing or insufficient.

### 9.5. Equivalence margin

The ±2 mm equivalence margin is a predefined analysis parameter in V11.

A statistical equivalence result does not, by itself, establish clinical interchangeability. The margin must be justified in relation to the intended application and the magnitude of clinically meaningful facial motion.

### 9.6. Filtering

Kinematic estimates, especially velocity and acceleration, depend on the selected filter cut-off frequency.

The sensitivity analysis is intended to characterize this dependence and should not be interpreted as proof that a particular cut-off perfectly preserves the biological signal.

## 10. Reproducibility

For reproducibility, users should:

1. Use the same Vicon export format.
2. Preserve the pairing between dental splint and headband recordings.
3. Verify marker names and reference marker availability.
4. Record the Python and package versions.
5. Keep the configuration parameters unchanged when reproducing a specific analysis.
6. Retain the generated CSV files and figures.
7. Record any modifications to the analysis script.

The current script contains local directory paths that must be adapted before execution.

The V11 pipeline is the reference implementation in this repository. Results from previous versions should not be assumed to be directly comparable without checking the corresponding processing parameters and analysis methods.

## 11. Citation

If this code is used in an academic publication, please cite the associated study:

Marcellin F, Rodriguez E, Raoult M, Le Moing AG, Colin E, Sarhan FR, Dakpé S.

*Comparative noise characterization of a 3D-printed headband and a maxillary dental splint as cranial references for marker-based facial motion capture.*

The bibliographic information should be updated once the manuscript has been accepted or published.

## 12. Data availability

The raw motion capture recordings are not included in this repository because they contain data collected from human participants.

Access to the data may be requested from the corresponding author, subject to the applicable ethical, regulatory and data-protection requirements.

## 13. Contact

**Félix Marcellin**  
INSERM UA-21 CHIMERE, Université de Picardie Jules Verne (UPJV), Amiens, France  
Institut Faire Faces, Amiens, France

GitHub repository:  
https://github.com/FelixMarcellin/Validation_diademe

---

**Pipeline version:** V11  
**Main script:** `comparaison_repere_V11_reperes faciaux_simplifiee.py`  
**Last documentation update:** November 2026
