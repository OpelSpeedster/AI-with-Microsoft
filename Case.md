Customer Insurance Purchases Case Study Project Overview This project is
a complete machine learning case study focused on predicting whether a
customer will purchase an insurance product based on two key features:
their Age and Estimated Salary.

The goal is to systematically evaluate several standard classification
algorithms, select the optimal model (the Support Vector Machine), and
use its decision boundary to generate actionable business insights and
make specific predictions for new customer profiles.

Table of Contents Dataset & Features

Methodology & Algorithms

Model Performance Summary

Key Insights & Hypotheses

Key Takeaways & Real-Life Applications

1\. Dataset & Features The study uses the Social_Network_Ads.csv
dataset. The data was processed through standard preparation steps:

Feature Name Role Description Age Independent Variable (X) Customer\'s
age in years. Estimated Salary Independent Variable (X) Customer\'s
estimated annual salary. Purchased Target Variable (y) Binary class: 0
(Did Not Purchase) or 1 (Purchased).

Export to Sheets Data Preparation Steps Splitting: The dataset was split
into an 80% Training Set and a 20% Test Set.

Scaling: StandardScaler was applied to standardize the features, which
is critical for distance-based algorithms like KNN and SVM to perform
correctly.

2\. Methodology & Algorithms Five common classification algorithms were
trained and evaluated on the scaled training data.

Algorithms Evaluated Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM) (with an RBF kernel)

Decision Tree Classifier (with entropy criterion)

Random Forest Classifier

Optimal Model Selection The F1-Score was used as the primary metric to
select the best model, as it provides the most balanced measure between
Precision and Recall.

3\. Model Performance Summary The performance metrics for all five
models on the test set are summarized below:

Classifier Accuracy Precision Recall F1-Score Logistic Regression 0.8625
0.9048 0.6786 0.7755 KNN 0.9125 0.8621 0.8929 0.8772 Support Vector
Machine 0.9250 0.8438 0.9643 0.9000 Decision Tree 0.8375 0.7778 0.7500
0.7636 Random Forest 0.8750 0.8000 0.8571 0.8276

Export to Sheets The Support Vector Machine (SVM) was selected as the
optimal model with the highest F1-Score of 0.900.

4\. Key Insights & Hypotheses The decision boundary of the optimal SVM
model revealed powerful insights into customer behavior.

Tested Hypotheses Hypothesis Analysis Salary has a stronger impact than
Age. Strongly Supported. The decision boundary is more horizontal than
vertical, meaning a change in salary is more likely to cross the
purchase threshold than a similar change in age. Older, high-salary
individuals are less inclined to purchase. Not Supported. The top-right
quadrant (older age, high salary) is solidly in the \"Purchased\"
(green) zone, showing they are the most likely group to buy. Younger,
high-salary individuals are more likely to purchase. Supported. The
decision boundary shows that as salary increases, the required age to
fall into the \"purchase\" zone decreases.

Export to Sheets Predictions for New Profiles The SVM model was used to
predict purchase behavior for new profiles (salaries ≥ \$200,000 were
capped for reliability).

Profile Age Estimated Salary Prediction G1.1 30 \$87,000 Will NOT
Purchase G1.3 40 \$100,000 WILL Purchase G1.4 50 \$0 WILL Purchase G2.1
18 \$0 Will NOT Purchase

Export to Sheets 5. Key Takeaways & Real-Life Applications Lessons
Learned Feature Scaling is Crucial: For distance-based models like SVM
and KNN, feature scaling (StandardScaler) is not optional; it is
required to prevent features with larger numerical ranges (like salary)
from dominating the distance calculations.

Visualizing the Boundary: Plotting the decision boundary provides a
powerful, intuitive understanding of the model\'s logic and the relative
importance of features, which is often more valuable for business
stakeholders than raw metrics.

Real-Life Applications The classification techniques employed are widely
applicable for predictive business scenarios:

Targeted Marketing Campaigns (Financial Services):

Goal: Optimize marketing spend by identifying customer segments most
likely to respond to a new premium product (e.g., a high-tier credit
card).

Model Use: A Random Forest Classifier could be trained on features like
account balance, credit score, and recent transaction history to predict
the binary outcome: Will Accept Offer or Will Not Accept Offer.

Customer Churn Prediction (Telecom/SaaS):

Goal: Proactively intervene to retain subscribers who are at high risk
of canceling their service.

Model Use: Logistic Regression (for its high interpretability) or an SVM
could be trained on metrics like frequency of support calls, recent dips
in usage, and contract type to predict the binary outcome: Likely to
Churn or Likely to Stay.
