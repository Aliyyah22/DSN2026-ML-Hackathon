# DSN2026-ML-Hackathon
Machine learning solution for the DSN AI Bootcamp Qualification Hackathon 2026 (ML Track), predicting product-store sales using a stacked ensemble of XGBoost, LightGBM, and CatBoost with target encoding.

## Project Overview

This project follows a full ML workflow: data understanding, cleaning, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, target encoding, stacking ensembles, error analysis, and final model selection.

## Final Model

- **Stacking ensemble**: XGBoost + LightGBM + CatBoost (all GridSearchCV-tuned), combined with a Ridge meta-model
- **Key engineered feature**: K-Fold target encoding on `product_code`
- **Full-data 5-fold CV RMSE**: 1075.60 (±19.52)
- **Kaggle Public Leaderboard Score**: 1070.28592

## Process Summary

- Investigated and resolved missing values in `store_size` and `product_weight_kg` using domain-driven, relationship-based imputation rather than simple mean/mode fills
- Compared multiple models (Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost)
- Tuned hyperparameters using GridSearchCV, validated against an untouched holdout set
- Identified and corrected a subtle cross-validation leakage issue in my target encoding methodology
- Conducted error analysis to understand systematic prediction patterns
- Tested multiple ensemble compositions and meta-models before finalizing my approach

## Files

- `DSN2026 Project.ipynb` — Full project notebook, including data exploration, modeling, tuning, and final submission generation

## Author

Aliyyah Adebayo
