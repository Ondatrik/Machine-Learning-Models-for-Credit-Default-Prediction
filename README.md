# Machine Learning for Credit Default Prediction

This project applies machine learning and statistical learning techniques, including Logistic Regression, Decision Trees, Random Forests, XGBoost and Feedforward Neural Networks, to predict credit defaults using historical loan data from the Indian online lending platform LoanTap. The repository demonstrates an end-to-end workflow for a binary classification problem, covering data cleaning, exploratory data analysis, feature engineering, model development, hyperparameter tuning and evaluation.

## Dataset

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/yekahaaagayeham/loantap-data?select=LoanTapData.csv

Repository Files:

* **Data Cleaning.ipynb** - initial data exploration, variable selection and data cleaning
* **Exploratory Data Analysis.ipynb** - in-depth EDA aimed at identifying data preprocessing steps
* **Data Preprocessing.ipynb** - data preprocessing and feature engineering
* **Logistic Regression.ipynb** - Logistic Regression model training, validation and testing
* **Decision Tree.ipynb** - Decision Tree classifier training, validation and testing
* **Random Forest.ipynb** - Random Forest classifier training, validation and testing
* **XGBoost.ipynb** - XGBoost classifier training, validation and testing
* **Feedforward Neural Network.ipynb** - FNN classifier training, validation and testing

## Model Performance

The models were evaluated using the F1 score, which was selected as the primary metric due to the class imbalance in the dataset.

| Model | Test F1 | Notes |
|-------|--------:|-------|
| Logistic Regression |  0.424 | Baseline linear model |
| Decision Tree |  0.416 | Entropy criterion |
| Random Forest |  0.426 | Best ensemble tree model |
| XGBoost | **0.436** | Best overall performance |
| Feedforward Neural Network | 0.428 | Two hidden layers (64, 32) |

## Conclusion

Among the evaluated models, XGBoost achieved the best predictive performance with an F1 score of **0.436**, outperforming both traditional machine learning models and the Feedforward Neural Network. Increasing the complexity of the neural network architecture did not improve predictive performance, meaning that gradient-boosted trees are a more suitable choice for this structured credit risk dataset.
