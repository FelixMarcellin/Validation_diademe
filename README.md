# A 3D-Printed Headband as a Non-Invasive Cranial Reference for Facial Motion Capture

## Overview

This repository contains the Python analysis pipeline developed to evaluate a 3D-printed headband as a non-invasive cranial reference system for marker-based facial motion capture.

The study compares facial kinematic measurements obtained simultaneously using two reference systems:

- **Maxillary dental splint:** a reference system anchored to the upper dentition.
- **3D-printed headband:** a wearable, non-invasive reference positioned over the fronto-parietal region.

The objective is to assess the extent to which the headband can provide usable facial kinematic measurements compared with the dental splint, while characterizing its geometric properties, temporal stability and measurement agreement.

The dental splint is used as a comparator, not as an error-free ground truth.

## Study design

The experimental study included:

- 9 healthy adult participants;
- 49 paired acquisition trials;
- 5 standardized facial movements (M1–M5);
- 105 reflective facial markers;
- 3 reference markers for each cranial reference system;
- A 10-camera Vicon optoelectronic motion capture system;
- A sampling frequency of 100 Hz.

Both reference systems were recorded simultaneously, allowing paired frame-by-frame comparisons under identical acquisition conditions.

## Analysis pipeline

The main analysis script is:

`comparaison_repere_V12_4_1_intra_participant.py`

The pipeline performs the following operations.

### 1. Data import and quality assessment

- Import paired Vicon recordings.
- Identify corresponding dental-splint and headband acquisitions.
- Match common facial marker trajectories.
- Reconstruct short missing trajectory segments using linear interpolation.
- Identify the longest continuous valid segment for each marker.
- Apply a minimum trajectory-duration criterion.

All analysable marker–trial observations are retained in the primary kinematic analysis, irrespective of movement amplitude.

### 2. Reconstruction of cranial reference frames

A separate local coordinate system is reconstructed for each device using its three reference markers.

The Kabsch algorithm, based on singular value decomposition, is used to estimate the frame-by-frame rigid transformation.

Facial marker trajectories are then expressed independently in the dental-splint and headband coordinate systems.

### 3. Facial kinematic analysis

The pipeline calculates and compares:

- Three-dimensional displacement;
- Velocity;
- Acceleration;
- Three-dimensional range of motion (ROM);
- Absolute and normalized errors;
- Correlation between kinematic signals;
- Angular deviations.

Trajectories are filtered using a second-order Butterworth low-pass filter with a 6 Hz cutoff frequency. A sensitivity analysis evaluates the influence of alternative cutoff frequencies.

### 4. Reference-system geometry and stability

The geometric properties of each reference-marker triangle are characterized using:

- Inter-marker distances;
- Triangle height;
- Triangle area;
- Temporal variability of inter-marker distances;
- Kabsch rigid-fit residual RMSD.

Relative translation and rotation between the headband and dental-splint coordinate systems are also evaluated throughout each acquisition.

These measurements quantify relative drift and must not be interpreted as absolute headband movement, since residual movement of the dental splint may also contribute.

### 5. Anatomical and amplitude-dependent analyses

Facial markers are mapped to 32 predefined anatomical zones.

The pipeline provides descriptive analyses of:

- Kinematic agreement across anatomical regions;
- Right–left comparisons where bilateral anatomical counterparts are available;
- Movement-specific agreement;
- Relationships between movement amplitude and measurement error.

Relative agreement metrics are restricted to the secondary subset of jointly moving observations. Absolute error metrics are also reported across the full analysable dataset.

### 6. Statistical analysis

The statistical analyses include:

- Descriptive statistics (median and interquartile range);
- Bland–Altman analyses;
- Mixed-effects variance-components models for repeated observations;
- Two one-sided tests (TOST) for the mean ROM difference;
- Participant-cluster bootstrap confidence intervals;
- Sensitivity analyses.

The exploratory ±2 mm TOST margin is an analytical reference value, not a clinically validated acceptance threshold. Equivalence of the mean bias does not establish individual-level interchangeability; the limits of agreement must also be considered.

## Main findings

The results indicate that:

- The headband provides usable displacement-based facial kinematic measurements in the experimental setting.
- Agreement is generally better for displacement than for velocity and acceleration.
- Translational stability varies between participants and acquisition trials.
- The geometric properties of the reference-marker triangle and the coupling of the device to the head are distinct aspects of measurement performance.
- The available results do not establish individual-level interchangeability with the maxillary dental splint.

Three headband size variants were developed after the acquisitions as a potential design improvement. These variants were not evaluated in the present study.

## Repository structure

The principal analysis script is:

```text
comparaison_repere_V12_4_1_intra_participant.py
```

The analysis generates tables and figures describing the different stages of the validation pipeline, including:

- Global kinematic agreement;
- Movement-specific comparisons;
- Anatomical-zone analyses;
- Reference-system geometry;
- Relative translation and rotation;
- Bland–Altman analyses;
- Intra-participant and intra-movement amplitude–error analyses;
- Participant-level bootstrap results;
- Filter sensitivity analyses.

The detailed output filenames are generated by the analysis script.

## Requirements

The analysis was developed using Python 3.9.

The main scientific Python dependencies are:

```text
numpy
pandas
scipy
matplotlib
seaborn
```

Install the dependencies with:

```bash
pip install numpy pandas scipy matplotlib seaborn
```

## Running the analysis

1. Clone or download this repository.
2. Install the required Python dependencies.
3. Prepare the paired Vicon CSV recordings according to the naming convention and input structure expected by the script.
4. Update the input and output directory paths in the script configuration.
5. Run the analysis:

```bash
python comparaison_repere_V12_4_1_intra_participant.py
```

The script will process the paired recordings and generate the corresponding statistical tables and figures in the configured output directory.

**Important:** The input directory structure, file-naming convention and marker labels must be consistent with the configuration expected by the script.

## Data availability

Raw motion-capture recordings are not included in this repository.

The recordings contain participant-derived motion data and are subject to the applicable ethical and data-protection requirements. Access to the underlying data should be requested from the corresponding author, subject to the relevant institutional and ethical approvals.

## Limitations

The findings should be interpreted in light of several limitations:

- The sample consisted of nine healthy adult participants.
- The study did not include patients with facial palsy.
- The dental splint was a comparator rather than an absolute rigid ground truth.
- Relative device drift cannot be attributed exclusively to the headband.
- The study was conducted under controlled experimental conditions using standardized facial movements.
- The three subsequently developed headband sizes were not evaluated.
- Clinical acceptability thresholds for ROM differences have not yet been established.

Further studies are required to assess fit, repositioning repeatability, inter-operator reproducibility, comfort and performance across different cranial morphologies and clinical populations.

## Reproducibility

The scripts are provided to document the processing and statistical analyses used in the associated scientific study.

For reproducibility, the Python version, input data structure, marker configuration and analysis parameters should be kept consistent with those described in the manuscript.

## Associated publication

**Marcellin F. et al.**

*A 3D-printed headband as a non-invasive cranial reference for facial motion capture: a comparative evaluation against a maxillary dental splint.*

Manuscript submitted to / prepared for **Scientific Reports**.

## Contact

For questions about the analysis pipeline, methodology or the headband design:

**Félix Marcellin**  
INSERM UA-21 CHIMERE, Université de Picardie Jules Verne (UPJV), Amiens, France  
Institut Faire Faces, Amiens, France  
Email: felix.marcellin@u-picardie.fr
