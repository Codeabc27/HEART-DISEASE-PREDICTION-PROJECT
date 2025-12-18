#  Heart Disease Prediction System  
### Machine Learning Project | Healthcare Analytics

##  Executive Summary
The Heart Disease Prediction System is a machine learning–based healthcare analytics project designed to predict the likelihood of heart disease using patient medical attributes. The project demonstrates an end-to-end implementation of a supervised learning workflow, including data preprocessing, feature scaling, model training, and evaluation. It highlights the practical application of machine learning in healthcare decision support systems and is suitable for academic, internship, and portfolio purposes.

##  Project Objectives
- Analyze patient medical data to identify patterns related to heart disease  
- Build an accurate and interpretable machine learning classification model  
- Apply data preprocessing, feature scaling, and model evaluation techniques  
- Demonstrate real-world use of machine learning in the healthcare domain  

##  Problem Statement
Cardiovascular diseases are among the leading causes of mortality worldwide. Early detection is critical for effective treatment. This project uses machine learning techniques to predict heart disease risk at an early stage, enabling data-driven and proactive medical decision-making.

## 🛠️ Technologies & Tools
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Machine Learning (Logistic Regression)  
- Git & GitHub  

##  Dataset Information
The project uses a realistic synthetic dataset containing 1000 patient records with 13 medical features and one target variable. The target variable indicates the presence or absence of heart disease.

### Dataset Features
| Feature | Description |
|--------|------------|
| age | Age of the patient |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | 1 = Heart Disease, 0 = No Disease |

##  Machine Learning Approach
A Logistic Regression model is used for binary classification due to its simplicity, efficiency, and high interpretability, which is essential for healthcare-related applications.

## Model Evaluation
The model performance is evaluated using accuracy score, confusion matrix, and classification report, including precision, recall, and F1-score to ensure reliable predictions.

##  How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction.git
