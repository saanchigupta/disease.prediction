"""
# Disease Prediction Using Logistic Regression

##  Introduction
Diabetes is a widespread health concern globally, marked by high blood sugar levels due to insufficient insulin production or resistance to insulin. Early detection is critical for effective management and prevention of severe complications. This project demonstrates how logistic regression can be used for binary classification to predict diabetes, based on the Pima Indians Diabetes Dataset.

## 📊 Dataset Overview
The dataset used is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases and includes data from 768 female patients of Pima Indian heritage. The dataset consists of the following features:

- **Pregnancies**: Number of times the individual has been pregnant  
- **Glucose**: Plasma glucose concentration after an oral glucose tolerance test  
- **Blood Pressure**: Diastolic blood pressure (mm Hg)  
- **Skin Thickness**: Triceps skin fold thickness (mm)  
- **Insulin**: 2-hour serum insulin (mu U/ml)  
- **BMI**: Body Mass Index (weight in kg/(height in m)^2)  
- **Diabetes Pedigree Function**: A function that scores the likelihood of diabetes based on family history  
- **Age**: Age in years  
- **Outcome**: Class label (0 = non-diabetic, 1 = diabetic)

##  Methodology

### 1. Data Preparation
- Imported required libraries such as `pandas`, `numpy`, and `sklearn`
- Loaded the dataset using `pandas.read_csv()`
- Split the dataset into training (75%) and testing (25%) sets

### 2. Feature Selection
- Applied Recursive Feature Elimination (RFE) to identify significant features
- Selected key features: Glucose, BMI, Age, Pregnancies, and Diabetes Pedigree Function

### 3. Model Building
- Used `LogisticRegression` from `sklearn.linear_model`
- Trained the model on the training dataset
- Performed predictions on the test dataset

### 4. Model Evaluation
- Evaluated the model using metrics such as Accuracy, Precision, Recall, and F1-score
- Applied cross-validation to ensure the robustness of the model

## Results
- **Accuracy**: Achieved accuracy 87.93%


##  Discussion
Logistic regression is a well-established algorithm for binary classification problems and offers a good balance of simplicity and performance. The model’s effectiveness depends on proper feature selection, handling of missing data, and normalization. The Pima Indians Diabetes Dataset serves as a strong foundation to demonstrate the application of logistic regression in healthcare prediction tasks.

## Conclusion
This project illustrates the effectiveness of logistic regression in predicting the likelihood of diabetes using patient data. The approach not only achieves satisfactory accuracy but also provides valuable insights into the most influential health indicators related to diabetes. This methodology supports the development of tools for early intervention and healthcare planning.
"""
