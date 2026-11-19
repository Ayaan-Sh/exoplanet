# Auto-generated results report

- Open Exoplanet Catalogue: 5,414 rows x 25 columns
- HYG star database (selected columns): 119,626 rows x 11 columns
- Catalogue `list` column (status) values: Confirmed planets=5287; Controversial=100; Retracted planet candidate=12; Solar System=9; Kepler Objects of Interest=5; Planets in binary systems=1
- Missing: age 100%, mass 49%, radius 23%, eccentricity 60%
- Duplicate planet names: 0
- HYG hides 'unknown distance' as a sentinel value 100000 pc: 10,225 stars affected (8.5%). Left as-is, it would wreck any distance statistics.
- Status counts: Confirmed=5288, Controversial=100, Retracted=12, Solar System=9, Candidate (KOI)=5
- Removed 117 non-confirmed objects (controversial, retracted, candidates) and set aside 9 Solar System bodies as reference points.
- Dropped 0 confirmed planets with no discovery year; parsed `lastupdate` to datetime (1 unparseable/missing).
- Converted RA/Dec text to degrees. Missing coordinates: 0
- Dropped sparse columns (> 75% missing): ['periastron', 'longitude', 'ascendingnode', 'age', 'hoststar_age']
- Planets above the 13 Jupiter-mass deuterium-burning limit (may really be brown dwarfs): 74 (flagged, kept)
- Orbital-period outliers flagged (log-IQR): 281 (longest 22,012 years). They are real directly-imaged planets, so flagged not removed.
- Size class assigned for 100% of planets (1152 of them estimated from mass because radius was missing).
- Planets in the (estimated) habitable zone: 70; small/rocky ones (HZ candidates): 13
- Validation: computed equilibrium temperature vs reported temperature (n=1331): median ratio = 1.07 (1.0 = perfect agreement)
- Skewness of radius: 2.22 (raw) -> 0.71 (log10)
- Host names parseable to a catalogue ID: 17% of planets ({'hd': 718, 'gl': 131, 'hip': 74})
- Naive join on star proper name would match only 7 planets, the ID-parsing approach below does far better.
  - HD: 691 of 718 planets matched to a star (96%)
  - HIP: 74 of 74 planets matched to a star (100%)
  - GL: 126 of 131 planets matched to a star (96%)
- Integrated: 889 of 5,288 planets enriched with star data (17%).
- Distance agreement between catalogues (n=881): median difference 1.5%, 25 planets differ by more than 20%.
- INSIGHT 1: 5,288 confirmed planets; the record year was 2016 with 1491 discoveries. 75% were found by the Transit method.
- INSIGHT 2: 44% of planets orbit their star in under 10 days. Most are 'hot' worlds because short orbits are the easiest to detect (selection bias), not because they are most common.
- INSIGHT 3: size classes: Neptune-size=1913, Jupiter-size=1669, Super-Earth=1213, Earth-size=478
- INSIGHT 4: host star classes: G=2207, K=1332, F=896, M=267, A=15, B=5
- INSIGHT 5: median distance 1,305 light-years; closest system Proxima Centauri at 4.2 ly.
- Grouped split by host star: train 3,250 / test 792 planets (shared hosts: 0)

Data quality before vs after:

| stage                |   rows |   columns |   missing_cells |   missing_% |
|:---------------------|-------:|----------:|----------------:|------------:|
| raw catalogue        |   5414 |        25 |           46236 |        34.2 |
| cleaned + integrated |   5288 |        49 |           68997 |        26.6 |
| model-ready (train)  |   3250 |         9 |               0 |         0   |

Radius prediction on unseen host stars:

| model             |    R2 |   RMSE (log10 R_earth) |
|:------------------|------:|-----------------------:|
| Linear Regression | 0.216 |                  0.33  |
| Random Forest     | 0.383 |                  0.292 |