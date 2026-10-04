# Turkish Crowdfunding Success Prediction

## Project Overview

This project predicts whether a crowdfunding project in Turkey will be **successful or unsuccessful** using machine learning.

The project compares **Logistic Regression** and **Random Forest** models, performs hyperparameter optimisation, addresses class imbalance, and evaluates model robustness using different train-test splits.

## Dataset

The project uses the `turkishCF.csv` dataset.

The dataset contains:

- **1,628 rows**
- **38 columns**

The target variable is:

- `başarısız` – Unsuccessful project
- `başarılı` – Successful project

The target is converted into numerical labels:

```text
başarısız → 0
başarılı  → 1
```

## Features

The dataset contains information related to:

- Project category
- Funding model
- Project-owner gender
- Location and region
- Campaign duration
- Promotional video availability and length
- Number of visuals
- Number of rewards
- Team size
- Website and social-media information
- Social-media follower counts
- Number of tags
- Description word count
- Funding target

## Preventing Target Leakage

Several variables are excluded from predictive modelling because they contain information that would only be available after or during the campaign.

Excluded variables include:

- `toplanan_tutar`
- `destek_orani`
- `destekci_sayisi`
- `yorumlar`
- `guncellemeler`

Identifiers, project names, owner names, raw descriptions, and raw project dates are also removed.

The campaign duration variable `gun_sayisi` is retained instead of using the raw start and end dates.

This ensures that the model attempts to predict campaign success using information that could realistically be available at the relevant prediction stage.

## Data Preprocessing

The following preprocessing steps are applied:

- Numerical missing values → **Median imputation**
- Categorical missing values → **Most-frequent imputation**
- Numerical variables → **Standardisation**
- Categorical variables → **One-Hot Encoding**
- Unknown test categories → Safely ignored using `handle_unknown="ignore"`

A **70:30 stratified train-test split** is used for the main experiment.

A separate **80:20 split** is also tested to evaluate robustness.

## Machine Learning Models

Two classification models are compared.

### 1. Logistic Regression

Logistic Regression is used as an interpretable baseline model.

The main hyperparameter tuned is:

```text
C
```

Five-fold stratified cross-validation is used with **ROC-AUC** as the optimisation metric.

The best Logistic Regression configuration is:

```text
C = 1
```

The best cross-validation ROC-AUC is approximately **0.8889**.

### 2. Random Forest

Random Forest is used because crowdfunding success may involve nonlinear relationships and interactions between features.

The following hyperparameters are tuned:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

The best configuration is:

```text
n_estimators = 150
max_depth = 15
min_samples_split = 5
min_samples_leaf = 1
max_features = sqrt
```

The best cross-validation ROC-AUC is approximately **0.9156**.

## Baseline Results

Using the 70:30 split:

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8589 | 0.7535 | 0.7683 | 0.5575 | 0.6462 | 0.8744 |
| Random Forest | 0.8630 | 0.7407 | 0.8286 | 0.5133 | 0.6339 | 0.8986 |

Random Forest provides the higher ROC-AUC, while Logistic Regression provides slightly better recall and F1-score in the baseline comparison.

## Addressing Class Imbalance

The dataset contains considerably more unsuccessful projects than successful projects.

Therefore, accuracy alone can be misleading.

The project uses:

```text
class_weight = "balanced"
```

to give greater importance to the minority class without generating synthetic observations.

The effect of class weighting is compared against the original tuned models.

## Class-Weighted Results

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.8589 | 0.7535 | 0.7683 | 0.5575 | 0.6462 | 0.8744 |
| Balanced Logistic Regression | 0.8057 | 0.7901 | 0.5584 | 0.7611 | 0.6442 | 0.8706 |
| Tuned Random Forest | 0.8569 | 0.7336 | 0.8028 | 0.5044 | 0.6196 | 0.8997 |
| Balanced Random Forest | 0.8364 | 0.7977 | 0.6260 | 0.7257 | 0.6721 | 0.9007 |

The **Balanced Random Forest** provides the highest ROC-AUC and F1-score among the tested models.

It also substantially improves recall for successful projects.

## Final Model

Based on the experimental comparison, **Balanced Random Forest** is selected because it achieves:

- Accuracy: **83.64%**
- Balanced Accuracy: **79.77%**
- Precision: **62.60%**
- Recall: **72.57%**
- F1-score: **67.21%**
- ROC-AUC: **90.07%**

The classification report shows that the model achieves a recall of approximately **72.57%** for successful projects.

This makes it a useful choice when identifying potentially successful crowdfunding projects is important.

## Model Robustness

The experiment is repeated using an **80:20 train-test split**.

### 80:20 Results

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8528 | 0.7595 | 0.7213 | 0.5867 | 0.6471 | 0.8808 |
| Random Forest | 0.8712 | 0.7574 | 0.8367 | 0.5467 | 0.6613 | 0.9073 |

Random Forest remains strong under the alternative split, with ROC-AUC increasing from approximately **0.8997** under the 70:30 experiment to **0.9073** under the 80:20 experiment.

## Error Analysis

The confusion matrix is used to examine:

- True positives
- True negatives
- False positives
- False negatives

For a crowdfunding screening system, **false negatives** can be particularly important because they represent potentially successful projects that are predicted as unsuccessful.

Therefore, a model with slightly lower overall accuracy but substantially better recall may be preferable depending on the practical objective.

The classification threshold could also be adjusted from the default 0.5 if the application requires a different precision-recall trade-off.

## Key Findings

- Random Forest generally achieves higher ROC-AUC than Logistic Regression.
- Hyperparameter tuning improves model selection.
- The dataset has meaningful class imbalance.
- Class weighting improves minority-class recall.
- Balanced Random Forest achieves the best ROC-AUC and F1-score among the tested candidate models.
- Random Forest remains robust when the train-test split is changed from 70:30 to 80:20.
- Preventing target leakage is important because several campaign outcome variables directly reveal the final success status.

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```text
turkish-crowdfunding/
│
├── Untitled1(3).ipynb
├── turkishCF.csv
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

3. Keep `turkishCF.csv` in the same directory as the notebook.
4. Open:

```text
Untitled1(3).ipynb
```

using Jupyter Notebook or JupyterLab.
5. Run the notebook cells from beginning to end.

## Conclusion

This project demonstrates how machine learning can be used to predict crowdfunding campaign success.

Both Logistic Regression and Random Forest are evaluated, with Random Forest providing stronger ROC-AUC performance.

After addressing class imbalance, **Balanced Random Forest** provides the best overall combination of ROC-AUC, balanced accuracy, recall, and F1-score among the tested models.

The results demonstrate that handling class imbalance and preventing target leakage are important when building a realistic crowdfunding success prediction system.
