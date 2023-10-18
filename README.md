SyriaTel Customer Churn Prediction

Business Understanding

Business Overview
SyriaTel, a prominent telecommunications company, aims to address revenue loss attributed to customer churn. The goal is to create a predictive model that determines whether a customer is likely to terminate their services. This constitutes a binary classification problem.

Audience
The primary stakeholders include SyriaTel's management and key decision-makers focused on reducing customer churn and enhancing revenue retention.

Business Objectives
1. Develop a classifier to predict customer churn.
2. Identify patterns or features indicative of potential churners.
3. Enable SyriaTel to implement targeted retention strategies.

Project Overview

This project focuses on predicting customer churn for SyriaTel by employing machine learning models and techniques. The process includes:

Data Preprocessing

- Loading the Dataset
- Handling Missing Values
- Feature Engineering
- Encoding Categorical Variables
- Removing Irrelevant Features

Exploratory Data Analysis (EDA)

- Univariate Analysis
- Analyzing Usage Patterns
- Investigating Demographics

Modeling

Logistic Regression Model

- A baseline model for its simplicity and interpretability.
- Results:
  - Accuracy: 85.16%
  - Precision: 62%
  - Recall: 5%
  - F1-Score: 9%

Random Forest Model

- A more complex model chosen for its ability to capture non-linear relationships and interactions.
- Results:
  - Accuracy: 95.65%
  - Precision: 92%
  - Recall: 68%
  - F1-Score: 78%

Tuned Random Forest Model

- Hyperparameters tuned for improved performance.
- Results:
  - Accuracy: 96.90%
  - Precision: 96%
  - Recall: 67%
  - F1-Score: 79%

Naive Bayes and K-Nearest Neighbors (KNN) Models

- Additional models providing complementary insights.

Observations and Conclusions

- Insights into customer segments with higher churn propensity.
- Key features influencing churn decisions identified.
- Model performance evaluated and discussed.

Recommendations

- Implement targeted retention strategies for identified high-risk customer segments.
- Enhance customer service during high-churn periods.
- Explore personalized offerings to incentivize customer retention.

Expected Impact

- Reduced customer churn leading to increased revenue retention.
- Enhanced customer satisfaction and loyalty.
- Strengthened competitive position in the telecom industry.

Getting Started

Dependencies

- Python 3.x
- Libraries: pandas, matplotlib, seaborn, scikit-learn

Usage

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `SyriaTel_Churn_Prediction.ipynb`.

Limitations and Recommendations

While the tuned Random Forest model demonstrates impressive performance, it's important to acknowledge certain limitations. Specifically, there may be subsets of customer records for which the model's predictions are less accurate. For instance, customers with unique usage patterns or demographics not well-represented in the training data may pose a challenge. Additionally, in a production setting, it's crucial to monitor the model's performance over time. As customer behavior evolves, the model may need periodic retraining or fine-tuning to maintain its accuracy. Furthermore, it's advisable to consider implementing a feedback loop where the model's predictions are reviewed by domain experts, allowing for continuous improvement and refinement of retention strategies. This iterative process ensures that the model remains a valuable asset in SyriaTel's efforts to reduce churn and enhance revenue retention.
