# Legendary Pokémon Classifier ⚡

## Overview
A fast, lightweight Supervised Machine Learning classification pipeline designed to predict whether a Pokémon is a "Legendary" based strictly on its raw base combat statistics (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed). 

The algorithm operates completely blind to lore, names, and typing, relying purely on the mathematical distribution of a species' combat potential.

## Technical Architecture
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Library:** Scikit-Learn
* **Algorithm:** Random Forest Classifier (`RandomForestClassifier`)
* **Preprocessing:** Standard Scaler (`StandardScaler`)

## Key Data Science Decisions
* **Feature Isolation:** Categorical and text columns (`Name`, `Type 1`, `Type 2`, `Generation`) were stripped to maintain a pure, continuous numerical feature matrix (`X`).
* **Mega Evolution Purge:** Megas, Primal Reversions, and Alternate Forms wildly skew base stat totals and create text-parsing edge cases (e.g., accidentally dropping *Meganium* or *Yanmega* when filtering the string "Mega"). This pipeline surgically purges all variants by dropping duplicate National Pokédex numbers (`#`), keeping only the baseline species for a pristine training environment.
* **Target Normalization:** The boolean target array (`Legendary`) was cast to binary integers (1s and 0s) for optimal Scikit-Learn compatibility.

## Pipeline Flow
1. **Ingest & Clean:** Filter standard Kaggle dataset to pure baseline species.
2. **Matrix Partition:** 80/20 Train-Test split.
3. **Scale:** Normalize all continuous stat distributions.
4. **Train:** Fit the Random Forest algorithm to the scaled numerical matrix.
5. **Inference:** Execute predictions on unseen test data to calculate final accuracy.