# 🌳 Decision Trees & Random Forest – Heart Disease Prediction

## 📌 Objective

This project is part of my **AI & ML Internship – Task 5**.  
The goal is to **build, visualize, and evaluate Decision Tree and Random Forest classifiers** to predict the presence of heart disease using the **Heart Disease dataset**.

---

## 🛠 Tools & Libraries Used

|Tool / Library|Purpose|
|---|---|
|**Python**|Core programming language|
|**Pandas**|Data loading, cleaning, wrangling|
|**NumPy**|Numerical computations|
|**Matplotlib / Seaborn**|Data visualization|
|**Scikit-learn**|Model building, preprocessing, evaluation|
|**Graphviz**|Decision tree visualization|
|**Google Colab**|Notebook execution environment|

---

## 🔄 Workflow – Step-by-Step Logic Flow

[Start]  
↓  
**Load dataset** (`pd.read_csv`)  
↓  
**Exploratory Data Analysis (EDA)** → shape, preview, missing values heatmap, target distribution  
↓  
**Preprocessing** → Label encoding for categorical features, feature scaling with `StandardScaler`  
↓  
**Split dataset** into Train/Test sets with stratified sampling  
↓  
**Decision Tree Classifier** → Train, visualize, tune `max_depth` to control overfitting  
↓  
**Accuracy vs Depth Analysis** → plot training & testing accuracy to pick best depth  
↓  
**Random Forest Classifier** → Train and compare accuracy with Decision Tree  
↓  
**Feature Importance** → Extract and plot top contributing features  
↓  
**Cross-validation** → Evaluate model robustness  
↓  
[End]

---

## 🧪 Steps Performed in Detail

### 1️⃣ Data Loading & EDA

- Dataset: **Heart Disease Dataset** (Kaggle)
    
- Checked shape, column info, statistical summary
    
- Visualized missing values heatmap
    
- Countplot for target distribution (heart disease presence vs absence)
    

### 2️⃣ Preprocessing

- **Label Encoding** → Converted categorical features to numeric form using `LabelEncoder`
    
- **Feature Scaling** → Applied `StandardScaler` to normalize all features except target
    

### 3️⃣ Splitting Data

- **Train/Test Split** → 80% training, 20% testing with `stratify=y` to maintain class balance
    

### 4️⃣ Decision Tree Model

- Trained a baseline `DecisionTreeClassifier`
    
- Tuned `max_depth` from 1 to 20
    
- Plotted **Accuracy vs Tree Depth** to visualize overfitting/underfitting
    
- Selected optimal depth with highest test accuracy
    
- Visualized final tree with `plot_tree`
    

### 5️⃣ Random Forest Model

- Trained `RandomForestClassifier`
    
- Compared accuracy with Decision Tree
    
- Extracted **feature importances** and plotted them using `sns.barplot`
    

### 6️⃣ Model Evaluation

- Used accuracy score for both models
    
- Performed 5-fold **Cross Validation** to assess generalization
    

---

## 📚 Vocabulary of Functions & Commands Used

|Command / Function|Purpose|
|---|---|
|`pd.read_csv(path)`|Load CSV dataset|
|`df.info()`|Overview of columns, types, and missing values|
|`sns.heatmap()`|Visualize missing data|
|`LabelEncoder()`|Encode categorical variables|
|`StandardScaler()`|Scale numerical features|
|`train_test_split()`|Split dataset into training and testing sets|
|`DecisionTreeClassifier()`|Initialize Decision Tree model|
|`.fit(X_train, y_train)`|Train model|
|`.score(X, y)`|Compute accuracy|
|`RandomForestClassifier()`|Initialize Random Forest model|
|`.feature_importances_`|Get importance score for each feature|
|`cross_val_score()`|Perform k-fold cross-validation|
|`plot_tree()`|Visualize decision tree|

---

## 📊 Key Insights

- **Decision Trees** can overfit at high depths — best performance found at depth ≈ optimal value from tuning.
    
- **Random Forests** generally outperform single trees due to bagging and averaging predictions.
    
- Feature importance analysis shows which health metrics most strongly influence heart disease predictions
