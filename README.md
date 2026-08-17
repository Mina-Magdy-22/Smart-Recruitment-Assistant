# 🚀 Smart Recruitment Assistant

## 📋 Overview
Modern hiring pipelines receive far more applications than teams can manually screen, making it easy for strong candidates to get lost in the volume. This project builds a **Machine Learning Classification System** that predicts whether a candidate is likely to advance in the hiring process. 

Using the **HR Analytics: Job Change of Data Scientists** dataset as a proxy, candidates actively looking to change jobs (`target = 1`) are treated as those worth prioritizing for outreach or advancement.

## 🎯 Approach
The project employs various classification models to find the best predictor of candidate intent. The pipeline involves:
- **Preprocessing:** Handling mixed data types (numeric, ordinal, and nominal features) using `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`, and `StandardScaler`.
- **Addressing Class Imbalance:** The dataset has a significant imbalance (~75% not looking, ~25% looking). `SMOTE` (Synthetic Minority Over-sampling Technique) is integrated into the training pipeline to mitigate this.
- **Modeling:** Training and comparing multiple models including Logistic Regression, Decision Tree, Random Forest, KNN, SVM, LightGBM, and XGBoost.
- **Evaluation:** Models are compared based on ROC-AUC, Precision, Recall, and F1-score to ensure the hiring team prioritizes the right candidates efficiently.
- **Deployment:** The best-performing model is fine-tuned using `GridSearchCV`/`RandomizedSearchCV` and exported via `joblib` for testing against unseen candidate data.

## 📊 Dataset
The project uses the `aug_train.csv` (19,158 records) and `aug_test.csv` (2,129 records) datasets.
### Key Features:
* `enrollee_id`: Unique ID for candidate
* `city`: City code
* `city_development_index`: Development index of the city
* `gender`: Gender of candidate
* `relevent_experience`: Relevant experience of candidate
* `enrolled_university`: Type of University course enrolled if any
* `education_level`: Education level of candidate
* `major_discipline`: Education major discipline of candidate
* `experience`: Candidate total experience in years
* `company_size`: Number of employees in current employer's company
* `company_type`: Type of current employer
* `last_new_job`: Difference in years between previous job and current job
* `training_hours`: Training hours completed
* `target`: 0 – Not looking for job change, 1 – Looking for a job change

## 🛠️ Tech Stack & Libraries
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Pipelines:** `scikit-learn`, `xgboost`, `lightgbm`, `imblearn`
* **Model Serialization:** `joblib`

## 📂 Notebook Structure
The workflow in the Jupyter Notebook is structured as follows:
1. `Imports` → Loading required libraries.
2. `Load Data` → Reading training and testing datasets.
3. `EDA (Exploratory Data Analysis)` → Understanding data distributions, unique values, and missing data patterns (e.g., `company_type` and `company_size` have ~30%+ missing values).
4. `Preprocessing` → Building the Scikit-Learn `ColumnTransformer` and Imbalanced-learn `Pipeline`.
5. `Modeling` → Training baseline models.
6. `Evaluation` → Comparing metrics and plotting ROC Curves & Confusion Matrices.
7. `Conclusion` → Exporting the best model.

## ⚙️ How to Run
1. Clone this repository.
2. Ensure you have the required datasets (`aug_train.csv` and `aug_test.csv`) in the root directory.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn joblib
   ```
4. Run the Jupyter Notebook to train the models and export the best pipeline.
