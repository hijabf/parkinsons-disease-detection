# Parkinson's Disease Detection Using Machine Learning

A machine learning project that detects Parkinson's Disease using biomedical voice measurements. Built using the Oxford Parkinson's Disease Detection Dataset.

## Overview

Parkinson's Disease affects up to 90% of patients through voice impairment, making voice analysis a non-invasive and cost-effective diagnostic tool. This project builds and compares multiple ML models to classify whether a person has Parkinson's Disease based on voice features.

## Dataset

- **Source:** Oxford Parkinson's Disease Detection Dataset (University of Oxford)
- **Features:** 23 biomedical voice measurements including jitter, shimmer, and harmonic ratios
- **Samples:** 197 voice recordings from 31 individuals (expanded with Parkinson's Telemonitoring Dataset)

## What I Built

- Loaded and explored the dataset using pandas
- Cleaned and preprocessed data with StandardScaler
- Handled class imbalance using SMOTE
- Built and compared 6 machine learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Neural Network (MLP)
- Performed hyperparameter tuning using GridSearchCV
- Evaluated models using Accuracy, Precision, Recall, F1-Score, and AUC
- Analyzed feature importance to identify key voice biomarkers
- Built a deployable Gradio interface for real-world usage

## Technologies Used

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- imbalanced-learn (SMOTE)
- Gradio

## How to Run

1. Clone the repository
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn gradio
   ```
3. Open the notebook in Google Colab or Jupyter Notebook
4. Run all cells

## Results

Multiple models were trained and compared. Random Forest and SVM achieved the highest performance. Feature importance analysis revealed that jitter and shimmer measurements are the most significant predictors of Parkinson's Disease.
