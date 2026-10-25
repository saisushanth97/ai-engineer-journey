# Customer Re-contact Prediction

## Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit following a marketing campaign. The goal is to identify customers who are more likely to respond positively so that campaign efforts can be targeted more effectively.

## Dataset

The project uses the Bank Marketing dataset containing 4,521 customer records and 17 features. The target variable `y` indicates whether the customer subscribed to a term deposit (`yes`/`no`).

The dataset contains customer demographics, financial information, previous campaign information, and details about the current campaign.

## EDA

Key observations from the exploratory analysis:

- The target variable is imbalanced, with approximately 88.5% `no` and 11.5% `yes`.
- `pdays = -1` indicates that the customer was not previously contacted in the previous campaign information represented in the dataset.
- Customer characteristics such as age, job, balance, and previous campaign information showed differences across target outcomes.
- `duration` was removed before model training because it represents the duration of the current call and would not be available at prediction time. Using it would introduce prediction-time leakage.

## Preprocessing

- The target variable was encoded as `0` and `1`.
- Numerical features were standardized using `StandardScaler`.
- Categorical features were converted into numerical representations using `OneHotEncoder`.
- A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.
- The preprocessing pipeline was fitted only on the training data and then applied to the test data.
- Stratified train/test splitting was used to preserve the target-class proportions.

## Models Compared

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Logistic Regression was used as the baseline model.

## Evaluation

Because the dataset is imbalanced, accuracy alone was not considered sufficient. Precision, recall, F1-score, ROC-AUC, and Average Precision (PR-AUC) were evaluated.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 88.84% | 55.17% | 15.38% | 24.06% | 0.737 | 0.328 |
| Decision Tree | 81.99% | 26.02% | 30.77% | 28.19% | 0.597 | 0.160 |
| Random Forest | 88.51% | 50.00% | 11.54% | 18.75% | 0.697 | 0.284 |
| Gradient Boosting | **88.95%** | **55.56%** | 19.23% | **28.57%** | **0.746** | **0.336** |

Gradient Boosting produced the strongest overall results among the evaluated models, although its improvement over Logistic Regression was modest.

## Threshold Analysis

The default classification threshold of 0.5 was also investigated.

For Gradient Boosting, lowering the threshold to 0.2 produced:

- Precision: 37.74%
- Recall: 38.46%
- F1-score: 38.10%

This demonstrates the trade-off between precision and recall when changing the classification threshold. The appropriate threshold ultimately depends on the business cost of false positives versus false negatives.

## Feature Importance

The Gradient Boosting model identified features related to previous campaign outcomes, customer age, and campaign/contact information as useful predictive signals.

Feature importance indicates that a feature contributes to prediction; it does not establish that the feature causes the target outcome.

## Final Conclusion

Gradient Boosting provided the best overall predictive performance among the models evaluated on this dataset. Its ability to capture nonlinear relationships and feature interactions likely helped it identify patterns that a linear model such as Logistic Regression could not capture as effectively.

However, the improvement was relatively modest, so the result should not be interpreted as proof that Gradient Boosting is universally better for this problem.

## Limitations

- The dataset contains relatively few observations compared with real-world banking datasets.
- The target variable is highly imbalanced.
- Model hyperparameters were not extensively optimized.
- Evaluation was based on a single train/test split.
- Threshold selection was exploratory and should ideally be performed using validation data while keeping the final test set untouched.
- Feature importance indicates predictive usefulness rather than causation.
- No external or production dataset was used for validation.

## Key Learnings

This project covered an end-to-end classical machine learning workflow:

`EDA → Data Leakage → Preprocessing → Train/Test Split → Classification → Model Comparison → Evaluation → Threshold Selection → Feature Importance`

The project also demonstrated why business objectives and class imbalance must be considered when evaluating classification models.
