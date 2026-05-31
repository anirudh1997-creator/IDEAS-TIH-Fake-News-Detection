# Fake News Detection and Evaluation with Confusion Matrix

## Project Overview
This repository contains the project submission for the **IDEAS - Institute of Data Engineering, Analytics and Science Foundation** Summer Internship Program 2026. 

The primary goal of this project is to build and evaluate an automated machine learning classification system capable of identifying and separating fake news articles from true news articles based on their textual content.

## Dataset Details
The dataset used for training and evaluation is sourced from a public Kaggle dataset consisting of two distinct categories:
* **True Articles:** Reputable, mainstream news content curated from `Reuters.com`.
* **Fake Articles:** Unreliable or intentionally fabricated articles flagged by platforms like Politifact and Wikipedia.
* **Content Focus:** The text corpus primarily focuses on political and world news.

## Workflow & Implementation Steps
The project is structurally implemented across 10 core milestones within the project notebook:
1. **Data Loading:** Importing datasets (`Fake.csv` and `True.csv`) into structured Pandas DataFrames.
2. **Label Generation:** Appending a binary `class` indicator column (`1` for Fake text, `0` for True text).
3. **Dataset Merging:** Concatenating the text sets and completely shuffling the rows to randomize training distribution.
4. **Data Feature Cleaning:** Dropping metadata variables (`title`, `subject`, `date`) to prevent data leakage and isolate core content text.
5. **Text Preprocessing:** Developing a dedicated pipeline function (`wordopt`) to uniform text to lowercase, strip hyperlinks/URLs, eliminate non-alphanumeric punctuation marks, and clear away redundant whitespaces.
6. **Train-Test Split:** Splitting the processed data matrices into a 75% training matrix and a 25% test matrix.
7. **Vectorization:** Converting the clean string sequences into a sparse matrix of numerical weights using `TfidfVectorizer`.
8. **Baseline Modeling:** Fitting a regularized `LogisticRegression` model and analyzing its performance via a classification report.
9. **Alternative Modeling:** Fitting a `DecisionTreeClassifier` to compare tree-based splits against linear separation.
10. **Hyperparameter Optimization:** Applying a 5-fold `GridSearchCV` cross-validation strategy to isolate the most robust values for regularization penalty strength (`C`) and the optimization algorithm (`solver`).

## Evaluation & Performance Metrics
Below are the training results and model parameters yielded by the implementation:

### 1. Logistic Regression Baseline
* **Precision:** [Insert your Precision score here, e.g., 0.99]
* **Recall:** [Insert your Recall score here, e.g., 0.99]
* **F1-Score:** [Insert your F1-Score here, e.g., 0.99]

### 2. Decision Tree Classifier
* **Test Accuracy:** [Insert your decimal score here, e.g., 0.9954]

### 3. Hyperparameter Tuning (GridSearchCV)
* **Optimal Parameters Discovered:** `[Insert best_params_ dictionary here, e.g., {'C': 10, 'solver': 'liblinear'}]`

## How to Run the Notebook
1. Clone this repository into your workspace environment.
2. Ensure you have the required source files (`Fake.csv` and `True.csv`) located in the root execution directory.
3. Install standard machine learning dependencies:
```bash
   pip install pandas scikit-learn
