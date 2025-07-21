# Bank Marketing Classification: SVM vs Random Forest

> A machine learning classification project comparing Support Vector Machines (SVM) and Random Forest to predict term deposit subscription outcomes using the UCI Bank Marketing Dataset. This project showcases data preprocessing, feature engineering, model optimization, and evaluation using precision, recall, and F1-score.

## 🧠 Project Summary

This project explores the performance of two powerful classification algorithms—**Support Vector Machines (SVM)** and **Random Forests**—in predicting whether a bank customer will subscribe to a term deposit, based on marketing campaign data.

The workflow includes comprehensive **data cleaning**, **feature encoding**, **hyperparameter tuning with GridSearchCV**, and performance evaluation using multiple metrics. The final comparison offers practical insights into model trade-offs in terms of runtime efficiency and predictive power.

---

## 📊 Tools & Technologies

- **Python** (Pandas, NumPy)
- **Scikit-learn** (SVM, RandomForest, GridSearchCV, metrics)
- **Jupyter Notebook**
- **Matplotlib & Seaborn** (for visualization)
- **OneHotEncoder, Label Encoding**
- **UCI Bank Marketing Dataset**

---

## 🧾 Dataset Overview

The dataset contains ~45,000 rows of client information and banking history. Key features include:

- **Client Info:** age, job, marital status, education
- **Banking Info:** balance, loan, housing
- **Contact & Campaign Info:** contact method, month, previous outcomes
- **Target Variable:** `y` (binary — subscribed: yes/no)

🔗 [Dataset Link - UCI Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## ⚙️ Workflow Overview

### 1. Data Preprocessing
- Parsed `.csv` with custom delimiter (`;`)
- Converted binary categorical variables to numeric (e.g., yes → 1, no → 0)
- Applied **OneHotEncoding** to nominal categorical variables
- Dropped low-impact features (`job`, `month`, `poutcome`, `contact`) based on correlation and feature importance analysis
- Removed duplicates and ensured clean column structure

### 2. Model Development

#### 🔍 Random Forest Classifier
- Built multiple models with varied parameters: `n_estimators`, `max_depth`, `min_samples_split`, and `criterion` (`gini`, `entropy`)
- Used **GridSearchCV** with 5-fold cross-validation
- Achieved best performance after feature pruning and tuning

#### 🔍 Support Vector Machine (SVM)
- Initial runs faced runtime constraints on large data (>12 hours in Colab)
- Strategically reduced sample size (2K–5K) for manageable training
- Tuned `C`, `kernel`, and `gamma` via GridSearchCV (with reduced `cv` to minimize compute)
- Performance improved after dropping low-impact features like `month`

---

## 📈 Model Evaluation

Each model was evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix Analysis**

### 📌 Best Performing Model
> ✅ **Random Forest (Model 3)**  
- Accuracy: **0.9086**  
- Precision: **0.5409**  
- Recall: **0.5502**  
- F1-Score: **0.502**  
- Demonstrated the best balance between predictive performance and runtime feasibility across the full dataset.

---

## 📉 Key Insights

- Random Forest consistently outperformed SVM on full dataset scale in both accuracy and stability.
- SVM showed promise in smaller data slices with higher precision in specific runs but was less reliable overall due to computational complexity.
- Dropping features like `job` and `month` improved performance and reduced training time.
- GridSearchCV significantly helped optimize performance through systematic hyperparameter exploration.

---

## 🧩 Future Improvements

- Explore **feature selection algorithms** to automate pruning
- Implement **SMOTE** or **class weighting** for imbalanced data handling
- Consider **XGBoost** or **LightGBM** as alternatives for faster, scalable modeling
- Add a live dashboard or Streamlit app for real-time predictions

