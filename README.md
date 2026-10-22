README — Pipeline de traitement des données
Validation d'un bandeau 3D imprimé comme référence crânienne pour la capture du mouvement facial
Ce dépôt contient le code de traitement des données utilisé dans l'article :

Marcellin F, Rodriguez E, Raoult M, Le Moing AG, Colin E, Sarhan FR, Dakpé S. Comparative noise characterization of a 3D-printed headband and a maxillary dental splint as cranial references for marker-based facial motion capture.

Le pipeline compare deux systèmes de référence crânienne utilisés en capture de mouvement facial : une gouttière dentaire maxillaire (dental splint) et un bandeau 3D imprimé (headband). Il produit toutes les figures et tous les tableaux de l'article, ainsi que le manuscrit Word final.

1. Structure du dépôt
text
.
├── comparaison_repere.py         # Pipeline principal (V7)
├── README.md                     # Ce fichier
│
├── DATA/                         # (non inclus dans le dépôt)
│   ├── gouttiere/                # Fichiers Vicon du dental splint (*_G_*)
│   └── diademe/                  # Fichiers Vicon du headband (*_D_*)
│
└── RESULTATS/                    # (généré par le pipeline)
    └── RESULTATS_COMPARAISON_V9_FIXED TRESHOLD 2.5/
        ├── 01_Marqueurs_Comparaison.csv
        ├── 02_Diademe_Stabilite.csv
        ├── ...
        ├── Fig_BlandAltman_ROM.png
        ├── Fig14_Rigidity_Diagnostics.png
        ├── ...
2. Prérequis
2.1. Environnement Python
Python ≥ 3.9

Dépendances principales :

text
numpy >= 1.21
pandas >= 1.3
scipy >= 1.7
matplotlib >= 3.4
seaborn >= 0.11
statsmodels >= 0.13     # pour Bland-Altman à effets mixtes
python-docx >= 0.8      # pour la génération du Word
Installe-les avec :

bash
pip install numpy pandas scipy matplotlib seaborn statsmodels python-docx
2.2. Données d'entrée
Les fichiers CSV doivent être exportés depuis Vicon Nexus au format standard :

Ligne 1 : commentaire / en-tête Vicon

Ligne 2 : fréquence d'acquisition (ex. 100.0)

Ligne 3 : noms des marqueurs (format X:Nom, Y:Nom, Z:Nom répété)

Lignes 4+ : données numériques, trois colonnes par marqueur (X, Y, Z)

Le pipeline attend deux répertoires :

DATA/gouttiere/ : fichiers *_G_*.csv (dental splint)

DATA/diademe/ : fichiers *_D_*.csv (headband)

Convention de nommage : chaque fichier doit contenir _G_ (splint) ou _D_ (headband) pour permettre l'appariement automatique. Exemple :

text
20230614_G_SMA_M1.csv    → splint, sujet SMA, mouvement M1
20230614_D_SMA_M1.csv    → headband, sujet SMA, mouvement M1
Les trois marqueurs de référence doivent être nommés M1, M2, M3 dans tous les fichiers. Les 105 marqueurs faciaux peuvent avoir n'importe quel nom, mais doivent être cohérents entre les fichiers splint et headband.

2.3. Structure attendue des fichiers
Marqueurs de référence : M1, M2, M3 (obligatoire)

Marqueurs faciaux : 105 marqueurs (F1–F7, S1–S10, N1–N4, Z1–Z11, Lc2, Lc5, Pps01, Pps02, etc. — voir le schéma de numérotation dans le dépôt)

Synchronisation : les deux flux sont acquis par le même système Vicon, donc frame à frame, sans recalage temporel nécessaire.

3. Configuration
Toute la configuration se fait en tête du fichier comparaison_repere.py, dans la section CONFIGURATION. Les paramètres sont regroupés par thème.

3.1. Chemins
python
DIR_G = Path(r"...\DATA\gouttiere\tous fichiers gouttiere")
DIR_D = Path(r"...\DATA\diademe\tous fichiers Diademe")
OUT   = Path(r"...\RESULTATS\RESULTATS_COMPARAISON_V9_FIXED TRESHOLD 2.5")
Adapte ces chemins à ta machine avant de lancer le pipeline.

3.2. Marqueurs de référence
python
REF_POINTS = ["M1", "M2", "M3"]
3.3. Marqueurs mobiles par mouvement
python
MOVING_MARKERS = {
    "M1": ["Lc2", "Lc5", "Pps01", "Pps02"],   # fermeture simple des paupières
}
Pour les mouvements non listés, tous les marqueurs sont considérés comme « moving ». Adapte ce dictionnaire à ta nomenclature si tu ajoutes des mouvements.

3.4. Filtrage et dérivation
python
FILTER_CUTOFF_HZ = 6.0       # fréquence de coupure Butterworth
FILTER_ORDER     = 2         # ordre (filtfilt, donc appliqué 2×)
EDGE_TRIM_FRAMES = 5         # frames rognées de chaque côté après dérivation
SWEEP_CUTOFFS_HZ = [2, 3, 4, 5, 6, 8, 10]   # balayage de coupure
3.5. Gestion des trous et de la qualité
python
MAX_GAP_FRAMES   = 10        # trous ≤ 10 frames interpolés linéairement
MIN_RUN_PCT      = 70.0      # % minimum de frames dans la plus longue séquence
MAX_INTERP_PCT   = 15.0      # % maximum de frames interpolées toléré
USE_QC_FOR_MAIN  = False     # exclure les essais QC="check" des résultats
3.6. Repère local
python
REPERE_MODE      = "kabsch"  # "kabsch" (recommandé) ou "m1m2m3"
BASELINE_FRAMES  = 5         # frames utilisées pour la baseline moyennée
ALIGN_ORIENTATION = True     # aligner les axes du headband sur le splint
"kabsch" : ajustement rigide optimal du triangle de référence sur les 3 marqueurs (distribue le bruit sur les 3 points → repère plus stable).

"m1m2m3" : construction stricte (origine M1, X = M1→M2, Z = normale au plan).

3.7. Repère cutané (cross-validation)
python
FIXED_THRESHOLD_MM  = 2.5    # seuil d'amplitude pour classer un marqueur "fixe"
MIN_FIXED_MARKERS   = 10     # nb minimum de marqueurs fixes pour un repère cutané
MIN_SKIN_FRAMES     = 30     # nb minimum de frames valides pour calculer l'amplitude
Un marqueur est fixe pour un mouvement donné si son amplitude peak-to-peak médiane dans le repère splint est inférieure à FIXED_THRESHOLD_MM. Seuil recommandé : 1.5 mm pour un repère cutané plus rigide (20/49 sessions valides) ou 2.5 mm pour un repère plus permissif (30/49 sessions valides). Le seuil de 2.5 mm est celui utilisé dans l'article principal.

3.8. Tolérances de stabilité
python
TRANSLATION_TOLERANCES_MM = [2.0, 3.0]
ROTATION_TOLERANCE_DEG    = 2.0
4. Lancement
4.1. Pipeline complet (figures + tableaux CSV)
bash
python comparaison_repere.py
Le script parcourt tous les fichiers splint, trouve le headband correspondant, analyse la paire, et produit :

RESULTATS/.../ : tous les CSV de synthèse

RESULTATS/.../*.png : toutes les figures de l'article

Durée estimée : 2–5 minutes pour 49 paires de fichiers à 100 Hz.

4.2. Génération du manuscrit Word
bash
python generate_article.py
Le script lit les figures produites et génère le document Word complet dans ~/Downloads/Article_Scientific_Reports_Headband_vs_Splint.docx.

Important : la photo des dispositifs (Fig01_devices.jpg) n'est pas générée par le pipeline. Elle doit être copiée manuellement dans le dossier RESULTATS/ avant de lancer generate_article.py.

5. Sorties
5.1. Fichiers CSV
Fichier	Contenu
01_Marqueurs_Comparaison.csv	Résultats par marqueur × essai (MAE, RMSE, Pearson, NRMSE, angles, etc.)
02_Diademe_Stabilite.csv	Dérive du headband dans le repère splint (translation + rotation)
03_Synthese_par_Mouvement.csv	Synthèse agrégée par mouvement
04_Synthese_par_Marqueur.csv	Synthèse agrégée par marqueur anatomique
06_Synthese_Globale_Cinematique.csv	Synthèse globale (moving, static, QC ok)
07_Stabilite_par_Mouvement.csv	Dérive par mouvement
08_Stabilite_par_Sujet.csv	Dérive par sujet
10_Bruit_Statique.csv	Bruit statique (position SD, velocity RMS, acceleration RMS)
11_Sensibilite_Coupure.csv	Sensibilité des métriques à la fréquence de coupure
11b_Sensibilite_Coupure_brut.csv	Données brutes du balayage de coupure
12_Couverture_Marqueurs.csv	Couverture et exclusions par marqueur
13_Marqueurs_Exclus.csv	Liste détaillée des marqueurs exclus
14_Rigidite_par_Fichier.csv	Rigidité du triangle par fichier × dispositif
15_Rigidite_Resume.csv	Résumé de la rigidité par dispositif
16_Par_Fichier_Moving.csv	Agrégation par fichier (anti pseudo-réplication)
17_Par_Fichier_Static.csv	Idem, marqueurs statiques
18_Skin_Fixed_Selection.csv	Liste des marqueurs fixes par session
19_Skin_Triple_Comparison.csv	Dérive croisée splint/headband dans le repère cutané
20_Skin_Summary_by_Movement.csv	Synthèse cutanée par mouvement
Fig_BlandAltman_ROM_stats.csv	Statistiques du Bland-Altman sur la ROM 3D
5.2. Figures
Fichier	Description	Figure article
Fig01_devices.jpg	Photo des dispositifs (à copier manuellement)	Figure 1
Fig02_NRMSE_by_Movement.png	NRMSE par mouvement	Figure 5
Fig06_Example_Norms.png	Trajectoire représentative	Figure 7
Fig08_Static_Noise_G_vs_D.png	Bruit statique splint vs headband	Figure 3
Fig10_Diadem_Drift_Series.png	Séries temporelles de dérive	Figure 8
Fig11_Stability_by_Movement.png	Dérive par mouvement	Figure 9
Fig12_Stability_by_Subject.png	Variabilité inter-sujet	Figure 10
Fig13_Peaks_vs_Cutoff.png	Pics absolus vs cutoff	Figure 4
Fig14_Rigidity_Diagnostics.png	Boxplots rigidité	Figure 2
Fig16_Skin_Drift_Comparison.png	Cross-validation cutanée	Figure 11
Fig_BlandAltman_ROM.png	Bland-Altman sur la ROM 3D	Figure 6
Fig01b_BlandAltman_Peaks.png	Bland-Altman sur les pics (supplementary)	S1
6. Méthodes
6.1. Construction du repère local
Deux constructions sont implémentées.

Mode m1m2m3 (strict) :

Origine : M1

Axe X : vecteur unitaire M1 → M2

Axe Z : produit vectoriel normalisé X × (M3 − M1)

Axe Y : Z × X

Mode kabsch (recommandé) :

Un triangle de référence est construit à partir des coordonnées moyennes de M1, M2, M3 sur les BASELINE_FRAMES premières frames valides.

Pour chaque frame, on estime la transformation rigide (rotation R, translation t) qui ajuste au mieux le triangle de référence sur les positions courantes (SVD, algorithme de Kabsch).

Le repère local est défini par cette transformation.

Cette construction distribue le bruit sur les 3 marqueurs au lieu de faire confiance à un seul point d'origine et à un seul axe principal.

Le mode kabsch est utilisé dans tous les résultats de l'article.

6.2. Baseline correction
Le déplacement est calculé par rapport à la moyenne des 5 premières frames valides (et non à la première frame seule, comme dans la version V6). Cela retire un décalage constant lié au bruit d'une seule frame, tout en préservant le déplacement dynamique.

6.3. Filtrage et dérivation
Filtrage passe-bas Butterworth zéro-phase (filtfilt), ordre 2, coupure 6 Hz par défaut.

Dérivation par gradient centré : vitesse = d(position)/dt, accélération = d(vitesse)/dt.

Suppression des bords : EDGE_TRIM_FRAMES frames retirées de chaque côté pour éviter les artefacts de bord du filtre.

6.4. Diagnostic de rigidité
Pour chaque dispositif et chaque session :

Distances inter-marqueurs M1-M2, M1-M3, M2-M3 (moyenne et écart-type temporel).

Hauteur du triangle par rapport à la base M1-M2.

Aire du triangle.

Kabsch RMSD : RMSD d'ajustement du triangle de référence sur les positions courantes. Mesure directe du bruit de position des marqueurs, indépendante de la géométrie du triangle.

Cette séparation entre bruit de marqueur (Kabsch RMSD) et bruit d'orientation (propagé par la géométrie) est le cœur de l'argument de l'article.

6.5. Bruit statique
Pour chaque dispositif :

On identifie les marqueurs qui ne bougent pas pendant un mouvement donné (amplitude < seuil).

On calcule leur écart-type de position, RMS de la norme de vitesse, RMS de la norme d'accélération.

Le ratio headband/splint donne le facteur de réduction du bruit.

6.6. Balayage de coupure
Pour chaque cutoff de SWEEP_CUTOFFS_HZ :

On recalcule les cinématiques (position, vitesse, accélération).

On enregistre le pic absolu de chaque norme pour chaque dispositif.

On calcule le Pearson, le RMSE normalisé, le NRMSE entre splint et headband.

Un dispositif dont le pic augmente fortement avec la coupure est dominé par le bruit haute fréquence ; un dispositif dont le pic se stabilise reflète un vrai signal.

6.7. Cross-validation cutanée
Pour chaque session :

Sélection des marqueurs fixes : marqueurs dont l'amplitude peak-to-peak médiane dans le repère splint est inférieure à FIXED_THRESHOLD_MM.

Construction du repère cutané : Kabsch optimal sur tous les marqueurs fixes (≥ MIN_FIXED_MARKERS).

Mesure de la dérive : on exprime le repère splint et le repère headband dans le repère cutané et on mesure leur dérive (translation + rotation).

Ratio headband/splint : si < 1, le headband dérive moins que le splint dans le repère cutané.

Cette cross-validation est indépendante du splint comme référence (elle utilise les marqueurs faciaux fixes) et permet de trianguler les résultats.

6.8. Bland-Altman
Deux Bland-Altman sont produits :

Sur la ROM 3D (figure principale) : differences = ROM_splint − ROM_headband, avec calcul du biais et des limites d'agrément (LoA = biais ± 1.96 × SD).

Sur les pics (supplementary) : même principe appliqué au pic de la norme du déplacement, avec un modèle à effets mixtes (sujet aléatoire + essai imbriqué).

6.9. Statistiques
Wilcoxon apparié : comparaison splint vs headband, agrégée par fichier (pas par marqueur) pour éviter la pseudo-réplication.

Kruskal-Wallis : comparaison des NRMSE entre mouvements.

Friedman : comparaison des NRMSE entre mouvements, apparié par sujet.

Bland-Altman à effets mixtes : pour les pics, avec statsmodels et repli classique si non-convergence.

7. Reproductibilité
7.1. Version des dépendances
Pour reproduire exactement les résultats de l'article, utilise les versions suivantes :

text
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
matplotlib==3.7.2
seaborn==0.12.2
statsmodels==0.14.0
python-docx==0.8.11
7.2. Paramètres exacts
Les résultats de l'article ont été produits avec :

text
FIXED_THRESHOLD_MM = 2.5
MIN_FIXED_MARKERS  = 10
REPERE_MODE        = "kabsch"
BASELINE_FRAMES    = 5
FILTER_CUTOFF_HZ   = 6.0
SWEEP_CUTOFFS_HZ   = [2, 3, 4, 5, 6, 8, 10]
7.3. Vérification
Pour vérifier que ton installation reproduit les résultats :

Lance comparaison_repere.py.

Ouvre 15_Rigidite_Resume.csv : tu dois retrouver

Splint : aire ≈ 405 mm², Kabsch RMSD ≈ 0.039 mm

Headband : aire ≈ 8553 mm², Kabsch RMSD ≈ 0.043 mm

Ouvre 20_Skin_Summary_by_Movement.csv : tu dois retrouver

Ratio translation D/G ≈ 0.97

Ratio rotation D/G ≈ 0.89

Vérifie que Fig_BlandAltman_ROM.png affiche un biais proche de −0.01 mm et des LoA de [−1.14, +1.13] mm.

Si ces valeurs diffèrent, vérifie les chemins, les noms de marqueurs et les paramètres.

8. Limitations connues
Le repère cutané n'est pas un gold standard absolu : la peau est déformable, et le RMSD d'ajustement médian de 0.77 mm reflète cette compliance intrinsèque.

Le seuil de sélection des marqueurs fixes (FIXED_THRESHOLD_MM) est un choix arbitraire. Les résultats ont été vérifiés robustes à 1.5 mm et 2.5 mm (voir Section 6.7 et Supplementary).

Le nombre de sessions pour la cross-validation cutanée est limité (30/49 avec un seuil de 2.5 mm, 20/49 avec un seuil de 1.5 mm), ce qui réduit la puissance statistique.

Le splint n'est pas un gold standard absolu : il peut micromouvoir pendant les contractions faciales forcées.

Le nombre de participants (9) est limité, et les tests inférentiels sont interprétés comme descriptifs plutôt que confirmatoires.

9. Citer ce travail
Si tu utilises ce code ou ces données dans un travail académique, merci de citer :

bibtex
@article{marcellin2026headband,
  title   = {Comparative noise characterization of a 3D-printed headband 
             and a maxillary dental splint as cranial references for 
             marker-based facial motion capture},
  author  = {Marcellin, F. and Rodriguez, E. and Raoult, M. and 
             Le Moing, A.G. and Colin, E. and Sarhan, F.R. and 
             Dakp{\'e}, S.},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Code and data available at 
             https://github.com/FelixMarcellin/Validation_diademe}
}
10. Contact
Pour toute question sur le pipeline, les données ou les résultats :

Félix Marcellin — felix.marcellin@u-picardie.fr

Laboratoire : INSERM UA-21 CHIMERE, Université de Picardie Jules Verne (UPJV), Amiens, France

Institut : Institut Faire Faces, Amiens, France

11. Licence
Ce code est distribué sous licence MIT (à adapter selon ton choix). Les données brutes ne sont pas incluses dans le dépôt pour des raisons de confidentialité (données de participants humains) ; elles peuvent être obtenues auprès de l'auteur correspondant sur demande raisonnable.

Dernière mise à jour : octobre 2026
Version du pipeline : V9 (seuil cutané 2.5 mm)
