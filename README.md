# Email Spam Classifier using Multinomial Naive Bayes

---

## Overview

This project implements a simple yet effective **Email Spam Classifier** using the **Multinomial Naive Bayes** algorithm. The classifier is trained on a dataset of Email messages labeled as either 'ham' (not spam) or 'spam' to learn patterns and accurately categorize new, unseen messages.

The key steps in this project include **data cleaning**, **text vectorization** using CountVectorizer, and training the classification model.

---

## Technologies Used

* **Python**
* **pandas** (for data manipulation)
* **numpy** (for numerical operations)
* **scikit-learn** (for machine learning model and text feature extraction)
    * `CountVectorizer`
    * `MultinomialNB` (Multinomial Naive Bayes)

---

## Dataset

The model is trained on a dataset contained in the `spam.csv` file (assumed to be available in the project directory). This dataset contains two columns:
* **Category**: The label, either 'ham' or 'spam'.
* **Text**: The content of the Email.

---

## Steps and Key Processes

The following steps were executed in the Jupyter Notebook (`Email_Spam_Classifier.ipynb`):

### 1. Data Loading and Initial Inspection
The data was loaded into a pandas DataFrame and inspected for its structure and contents.

### 2. Data Cleaning
* **Missing Values**: Checked for and confirmed there are **0** null values in either the 'Category' or 'Message' columns.
* **Duplicate Removal**: Identified **415** duplicate messages and removed them to ensure the model is trained on unique samples, resulting in 0 duplicates remaining.

### 3. Feature Engineering (Label Encoding)
A new numerical column, **`spam`**, was created by converting the categorical 'Category' column into binary integer labels:
* 'ham' is mapped to **0**
* 'spam' is mapped to **1**

### 4. Splitting Data
The data was split into **training** and **testing** sets for both messages (features, $X$) and their corresponding spam labels (targets, $y$).

### 5. Text Vectorization
The **`CountVectorizer`** (also known as Bag-of-Words) was used to transform the text messages into a matrix of token counts (numerical features). This step converts the raw text data into a format the machine learning model can process.
* `v.fit_transform(X_train)` was used to learn the vocabulary and create the numerical feature vectors for the training data.

### 6. Model Training
A **Multinomial Naive Bayes (`MultinomialNB`)** model, which is well-suited for classification with discrete features like word counts, was initialized and trained using the vectorized training data (`X_train_count`) and the training labels (`y_train`).

### 7. Prediction Example
The trained model was used to predict the category of two new sample messages:
1.  `"Hey lets meet tommorow"`
2.  `"20% discount on our website avail today"`

The model predicted:
* Message 1: **0** (Ham)
* Message 2: **1** (Spam)

---

## How to Run the Project

1.  **Dependencies**: Ensure you have Python and the required libraries installed:
    ```bash
    pip install pandas numpy scikit-learn
    ```
2.  **Files**: Make sure you have the following files in your project directory:
    * `Email_Spam_Classifier.ipynb` (The Jupyter Notebook with the code)
    * `spam.csv` (The dataset)
3.  **Execution**: Open the Jupyter Notebook and run the cells sequentially to reproduce the analysis and model training.

---

## Next Steps / Potential Enhancements

* Calculate and report the **accuracy** or other metrics (Precision, Recall, F1-Score) of the model on the test data (`X_test` and `y_test`).
* Implement a **Pipeline** to streamline the vectorization and model training process.
* Try a different vectorization technique, such as **TF-IDF**.
* Experiment with other classification algorithms like **Support Vector Machines (SVM)** or **Logistic Regression**.
