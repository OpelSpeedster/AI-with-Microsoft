# Project 1: Customer Insurance Purchases Case Study  

**Overview**: This project is a complete machine learning case study focused 
on predicting whether a customer will purchase an insurance 
product based on two key features: Age and Estimated Salary.

**Goal**: The goal is to systematically evaluate several standard classification
algorithms, select the optimal model (the Support Vector Machine), and
use its decision boundary to generate actionable business insights and
make specific predictions for new customer profiles.



1\. **Dataset & Features** : The study uses the Social_Network_Ads.csv
dataset. The dataset was split
into an 80% Training Set and a 20% Test Set.

StandardScaler was applied to standardize the features, which
is critical for distance-based algorithms like KNN and SVM to perform
correctly.

2\. **Methodology & Algorithms** : Five common classification algorithms were
trained and evaluated on the scaled training data.

Algorithms Evaluated Logistic Regression

- K-Nearest Neighbors (KNN)

- Support Vector Machine (SVM) (with an RBF kernel)

- Decision Tree Classifier (with entropy criterion)

- Random Forest Classifier

**Optimal Model Selection** : The F1-Score was used as the primary metric to
select the best model, as it provides the most balanced measure between
Precision and Recall.

3\. **Model Performance Summary** : The performance metrics for all five
models on the test set. The Support Vector Machine (SVM) was selected as the
optimal model with the highest F1-Score of 0.900.

