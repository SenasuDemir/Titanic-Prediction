# Titanic-Prediction

This project aims to predict the survival of Titanic passengers using machine learning models. Leveraging various classifiers, we analyze patterns in the Titanic dataset to understand which features contribute to survival rates and determine the best-performing model.

## Project Overview

The goal is to predict if a passenger survived the Titanic disaster based on specific features in the dataset, such as age, passenger class, and fare paid. This is a binary classification problem where the target variable `Survived` is either:
- `1`: Survived
- `0`: Did not survive

## Dataset

The Titanic dataset provides demographic and journey details of each passenger. Key columns include:
- `PassengerId`: Unique identifier for each passenger.
- `Pclass`: Passenger's class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Name`: Passenger's full name.
- `Sex`: Gender.
- `Age`: Age (some values may be missing).
- `SibSp`: Number of siblings/spouses aboard.
- `Parch`: Number of parents/children aboard.
- `Ticket`: Ticket number.
- `Fare`: Ticket fare paid in British Pounds.
- `Cabin`: Cabin number (may be missing).
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Project Structure

- **Data Preprocessing**: Data cleaning and feature engineering are done to handle missing values, encode categorical features, and normalize data where needed.
- **Model Training**: We train the following classifiers:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Decision Tree Classifier
  - Gaussian Naive Bayes
  - Bernoulli Naive Bayes
  - K-Neighbors Classifier
- **Model Evaluation**: Each model’s performance is evaluated using:
  - Classification report (precision, recall, F1-score).
  - Accuracy score.

## Performance

| **Model**                       | **Accuracy** |
|---------------------------------|--------------|
| Random Forest Classifier        | 0.832        |
| Gradient Boosting Classifier    | 0.821        |
| Logistic Regression             | 0.810        |
| Decision Tree Classifier        | 0.804        |
| Gaussian Naive Bayes            | 0.782        |
| Bernoulli Naive Bayes           | 0.782        |
| K-Neighbors Classifier          | 0.732        |


## Results

The highest accuracy of **83.2%** was achieved using the **Random Forest Classifier**, closely followed by the **Gradient Boosting Classifier** with **82.1%**. These results demonstrate the robustness of ensemble methods for this binary classification task.
