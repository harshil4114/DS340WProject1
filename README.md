## DS340W Project Credit and LoanRisk Prediction
DS340W – Data Science Capstone Project
Name - Credit and Loan Risk Prediction
Instructor: Professor Kaamran Raahemifar
Authors: Harshil Shinde & Samarth Patel

# Project Information
This project develops a comprehensive end-to-end credit-risk classification framework using the U.S. SBA FOIA Loan Dataset, implemented as part of DS340W under Professor Kaamran Raahemifar.
hat This Project Includes
✔ Full Machine-Learning Pipeline
Data cleaning (dates, NAICS codes, ZIP3, SBA share, etc.)
Feature engineering
One-hot encoding
Train/test split
Imbalance handling
Model training
Model comparison
Final evaluation


✔ Models Used
Logistic Regression
Random Forest
XGBoost
LightGBM
Neural Network (MLP)

✔ Modern Enhancements
Optuna hyperparameter tuning for XGBoost
SHAP interpretability
Feature importance plots
Advanced visualizations

📊 Visualizations Included

This project recreates all the major graphs shown in the parent research paper. These visualizations include starting-month versus count histograms, portfolio performance curves over time, bad-rate plots across months, and rolling bad-rate after one year. It also contains boxplots after removing outliers, violin and box-and-whisker plots comparing important variables to the risk outcome, pie charts showing distributions of borrower categories, and histograms of significant numeric variables. A full correlation heatmap was generated to show relationships between features.

Model-related visuals include normalized and raw confusion matrices for every model, an accuracy comparison bar chart, ROC curves for each model, a combined ROC curve for all models, and a single clean ROC curve for XGBoost alone. These match the style and purpose of Figures 17–25 from the parent paper. SHAP visualizations include summary plots and feature-importance bar charts, offering a modern extension to the original work.📊 Visualizations Included

This project recreates all the major graphs shown in the parent research paper. These visualizations include starting-month versus count histograms, portfolio performance curves over time, bad-rate plots across months, and rolling bad-rate after one year. It also contains boxplots after removing outliers, violin and box-and-whisker plots comparing important variables to the risk outcome, pie charts showing distributions of borrower categories, and histograms of significant numeric variables. A full correlation heatmap was generated to show relationships between features.

Model-related visuals include normalized and raw confusion matrices for every model, an accuracy comparison bar chart, ROC curves for each model, a combined ROC curve for all models, and a single clean ROC curve for XGBoost alone. These match the style and purpose of Figures 17–25 from the parent paper. SHAP visualizations include summary plots and feature-importance bar charts, offering a modern extension to the original work.

🚀 Tools and Technologies

The project was developed using Python in Google Colab. Key packages include Pandas and NumPy for data manipulation, Scikit-Learn for preprocessing and classical machine-learning models, XGBoost and LightGBM for boosting models, Optuna for Bayesian optimization, and SHAP for feature explainability. Matplotlib and Seaborn were used extensively to produce publication-quality graphs. All code is organized in a notebook for easy execution and reproducibility.
