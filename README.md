# Dry Bean Classification using Machine Learning

## a. Problem Statement

The aim of this project is to predict the variety of a dry bean grain from measurements of its shape and size. There are 7 bean varieties in the data, so this is a multi-class classification problem.

I trained five classifiers on the same training data (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) and compared them on the same test data using Accuracy, AUC, Precision, Recall, F1 and MCC. I also built a Streamlit app where test data can be uploaded and the saved models are evaluated on it.

## b. Dataset Description

I used the Dry Bean Dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The dataset was created by Koklu and Ozkan (2020). They photographed 13,611 bean grains of 7 registered varieties and extracted 16 features from each image.

- Number of instances: 13,611
- Number of features: 16 (all numeric)
- Target column: `Class`
- Classes: SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON
- Missing values: none

The features are Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness and ShapeFactor1 to ShapeFactor4.

The classes are not balanced. DERMASON is the largest class with 3,546 grains and BOMBAY is the smallest with 522 grains. Because of this I did not rely only on accuracy, and used macro-averaged precision, recall and F1 along with MCC.

| Class | Count |
|---|---:|
| DERMASON | 3546 |
| SIRA | 2636 |
| SEKER | 2027 |
| HOROZ | 1928 |
| CALI | 1630 |
| BARBUNYA | 1322 |
| BOMBAY | 522 |

I split the data into 80% training (10,888 rows) and 20% testing (2,723 rows) using a stratified split with random_state=42. The test set is saved as `test_data.csv` in the repository.

## c. GitHub Repository Link

GitHub: https://github.com/CodeWithN/ml-assignment-2

Streamlit app: https://ml-assignment-2-jxp8prgr7uve7ajh5vg7cv.streamlit.app

Files in the repository:

```text
ml-assignment-2/
|-- app.py
|-- train_models.py
|-- metrics_utils.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- data/
|   |-- dry_bean.csv
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    |-- metrics.csv
```

`train_models.py` trains and saves the models, `metrics_utils.py` calculates the six metrics and `app.py` is the Streamlit app. Each model is saved as a scikit-learn Pipeline, so the StandardScaler is saved inside the same file as the model.

## d. Models Used

Settings used for each model:

| Model | Settings |
|---|---|
| Logistic Regression | StandardScaler, max_iter=1000 |
| Decision Tree | max_depth=8, min_samples_leaf=5 |
| KNN | StandardScaler, n_neighbors=7 |
| Naive Bayes | GaussianNB, default settings |
| SVM | StandardScaler, RBF kernel, C=1.0 |

Since there are 7 classes, Precision, Recall and F1 are macro averages, and AUC is the one-vs-rest AUC averaged over all classes.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.9023 | 0.9837 | 0.9164 | 0.9132 | 0.9147 | 0.8819 |
| KNN | 0.9137 | 0.9858 | 0.9290 | 0.9245 | 0.9265 | 0.8956 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9949 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed very well with about 92% accuracy, almost the same as SVM. This shows that after scaling, most of the bean classes can be separated with linear boundaries. |
| Decision Tree | Got about 90% accuracy, which is lower than the other models except Naive Bayes. It made the most mistakes between SIRA and DERMASON. I limited the depth to 8 so that it does not overfit. |
| KNN | Gave about 91% accuracy. Scaling the features was important here because KNN uses distances. The model file is the biggest because KNN keeps all the training data. |
| Naive Bayes | Worst model with about 76% accuracy. Many features are strongly correlated with each other (for example Area and ConvexArea), so the independence assumption of Naive Bayes does not hold. It classified only 46% of BARBUNYA beans correctly and confused most of them with CALI. |
| Support Vector Machine | Best model, with the highest value in all six metrics (accuracy 92.2%, MCC 0.906). The RBF kernel can handle some non-linear boundaries that Logistic Regression cannot. It was the slowest model to train. |
| Overall, Winner for your dataset? | SVM is the winner for this dataset. Logistic Regression is very close behind and is a simpler model, so it would also be a good choice. |

In all models BOMBAY was classified 100% correctly, because these beans are much bigger than the others. Apart from Naive Bayes, most of the errors in the models were between SIRA and DERMASON, and between BARBUNYA and CALI, which look similar in size and shape.

## Streamlit App

The app has the following features:

- Upload a test CSV file (or use the sample `test_data.csv` given in the app)
- Dropdown to select one of the five models
- Shows Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Shows the confusion matrix and classification report
- Shows a comparison of all five models on the uploaded data

The models are only loaded from the `model` folder and are not trained again in the app.

## How to Run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
