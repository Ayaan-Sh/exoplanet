# Exoplanet Explorer: Data Wrangling Pipeline + Interactive Dashboard
**Course:** Data Wrangling and Preprocessing (DWP) - Mini Project (Internal Assessment)
**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib/Seaborn (analysis), Streamlit + Plotly (dashboard)

## 1. What is this project?
Astronomers have confirmed over 5,000 planets outside our Solar System. This project takes the raw public catalogue,
**cleans, transforms and integrates** it with a star catalogue, explores it, prepares it for machine learning, and shows
the result in an **interactive dashboard**.

Questions answered: How did discovery grow? What kinds of planets exist? Which orbit in the *habitable zone*?
Can we predict a planet's size from its orbit and star?

## 2. Datasets (public, no login, downloaded automatically by the code)
| Name | Link | Content |
|---|---|---|
| Open Exoplanet Catalogue | https://github.com/OpenExoplanetCatalogue/oec_tables (CSV: https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.txt) | 5,414 objects: mass, radius, orbit, host star, discovery method and year |
| HYG Star Database | https://github.com/astronexus/HYG-Database (CSV: https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv) | ~120,000 stars: spectral type, colour, luminosity, constellation, distance |

Background reading: NASA Exoplanet Archive https://exoplanetarchive.ipac.caltech.edu (the official source; the Open Exoplanet Catalogue is a community copy).

## 3. How to run
```bash
pip install -r requirements.txt
jupyter notebook Exoplanet_Wrangling_EDA.ipynb   # step-by-step wrangling (already executed)
python exo_pipeline.py                           # same pipeline as a script; creates data/processed/*
streamlit run app.py                             # opens the dashboard in your browser
```
The cleaned data is already included in `data/processed/`, so `streamlit run app.py` works immediately.
Internet is needed only the first time you re-run the pipeline.

## 4. Folder structure
```
Exoplanet_Wrangling_EDA.ipynb   wrangling + EDA + ML (main notebook, outputs visible)
exo_pipeline.py                 same code as a script (# %% cell markers)
app.py                          Streamlit dashboard
test_app.py                     automated dashboard test (filters, checkboxes)
data/raw/                       source files    data/processed/  clean data + wrangling log
outputs/figures/                7 charts        outputs/report.md  auto-generated numbers
.streamlit/config.toml          dark space theme
```

## 5. Wrangling steps (maps to the assignment topics)
| Stage | What we did | Why |
|---|---|---|
| Collection | Download two CSVs with `requests`, cache locally, keep only needed star columns | Reproducible; star file is 34 MB |
| Audit | Missing %, dtypes, duplicates, sentinel values | Find problems before fixing |
| Cleaning | Keep only **confirmed** planets (drop 117 controversial/retracted/candidates); set aside 9 Solar System bodies as a reference; parse dates; convert text coordinates ("19 19 43.4") to degrees; drop columns >75% missing; fix non-positive values; flag mass above the 13 Jupiter-mass limit (74 possible brown dwarfs); flag 281 period outliers with log-IQR; replace star-catalogue **sentinel distance 100000 pc** (8.5% of stars) with NaN | Wrong rows or fake values would corrupt every statistic |
| Transformation | Jupiter -> Earth units, light-years, density, **binning** into size classes (with mass fallback for 1,152 planets), star class by temperature, stellar luminosity, **habitable-zone** estimate, equilibrium-temperature cross-check, log features | Makes the data comparable and meaningful |
| Integration | The two catalogues share **no ID**. We parse the host name out of the planet name ("HD 209458 b" -> HD 209458) and match on HD / HIP / Gliese numbers. Matched 691/718 (HD), 74/74 (HIP), 126/131 (Gliese). Naive name join matched only 7 planets. Distances cross-checked between sources (median difference 1.5%) | Real-world data rarely has a clean join key |
| EDA | Discoveries per year, radius-period plot with our Solar System overlaid, HR diagram, distance, correlation | Understand and communicate patterns |
| ML preprocessing | **Grouped split by host star** (no star in both train and test), median imputation, scaling, one-hot encoding, all fitted on train only; baseline models | Avoids data leakage |

## 6. Dashboard
Sidebar filters (year, method, size class, distance, habitable-zone only, star-data only) drive every chart.
Tabs: **Overview** (discoveries per year, method share) | **Planet population** (radius vs period with Solar System overlay, mass-radius) |
**Host stars** (HR diagram, star classes, metallicity) | **Sky map** (Mollweide projection, nearest planets) |
**Habitable zone** (candidates table, planet profile card) | **Data quality** (the wrangling log) | **Data** (table + CSV download).

## 7. Key results (computed by the code, see `outputs/report.md`)
- 5,288 confirmed planets; record year 2016 with 1,491 discoveries; 75% found by the transit method.
- 44% orbit in under 10 days. This reflects **detection bias** (close-in planets are easiest to find), not what is typical.
- Size classes: Neptune-size 1,913 | Jupiter-size 1,669 | Super-Earth 1,213 | Earth-size 478.
- 70 planets sit in the estimated habitable zone; 13 are small enough to be rocky. Closest system: Proxima Centauri at 4.2 light-years.
- Radius prediction on unseen stars: Linear Regression R2 = 0.22, Random Forest R2 = 0.38.

## 8. Limitations (be ready to discuss)
- The habitable zone uses fixed flux limits (0.53 and 1.1 times Earth's), a simplification of published models. It says nothing about atmosphere or water.
- Only 17% of planets (889) could be matched to the star catalogue, because most hosts have survey names (Kepler-, K2-, TOI) that HYG does not contain.
- The model is weak (R2 0.38). Planet size depends on things not in the data (composition, atmosphere), and 1,152 size classes are estimated from mass.
- Computed vs reported equilibrium temperature has a median ratio of 1.07, so they agree only roughly (albedo and heat redistribution differ).
- The catalogue snapshot ends in 2023, so newer discoveries are missing. The NASA Exoplanet Archive has the latest list.

## 9. Possible extensions
Pull live data from the NASA Exoplanet Archive TAP API, cluster planets by properties (k-means), add Kepler/TESS-specific views, deploy the dashboard on Streamlit Community Cloud.
