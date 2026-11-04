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

I split the data into 80% training (10,888 rows) and 20% testing (2,723 rows) using a stratified split with random_state=2026. The test set is saved as `test_data.csv` in the repository and is never used for training or tuning.

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
    |-- best_params.csv
```

`train_models.py` tunes, trains and saves the models, `metrics_utils.py` calculates the six metrics and `app.py` is the Streamlit app. Each model is saved as a scikit-learn Pipeline, so the scaler (or power transform) is saved inside the same file as the model. `model/best_params.csv` records the hyperparameters chosen for each model.

## d. Models Used

Each model's hyperparameters were tuned with `GridSearchCV` using 5-fold stratified cross-validation on the **training split only**, with macro F1 as the score. The best setting was then refitted on the full training split and evaluated once on the test data.

| Model | Preprocessing | Grid searched | Selected | CV F1 (macro) |
|---|---|---|---|---:|
| Logistic Regression | StandardScaler (newton-cg solver, tol=1e-8) | C in {0.01, 0.1, 1, 10, 100} | C=100 | 0.9370 |
| Decision Tree | none | criterion in {gini, entropy}, max_depth in {6, 8, 10, 12, None}, min_samples_leaf in {1, 5, 10, 20} | gini, max_depth=10, min_samples_leaf=10 | 0.9207 |
| KNN | StandardScaler | n_neighbors in {3, 5, 7, 9, 11, 15, 21}, weights in {uniform, distance}, p in {1, 2} | n_neighbors=7, weights=distance, p=2 | 0.9362 |
| Naive Bayes (Gaussian) | none or PowerTransformer (searched) | transform, var_smoothing in {1e-9, 1e-7, 1e-5} | PowerTransformer, var_smoothing=1e-9 | 0.9021 |
| SVM (RBF kernel) | StandardScaler | C in {1, 10, 100}, gamma in {scale, 0.01, 0.1} | C=10, gamma=scale | 0.9427 |

Since there are 7 classes, Precision, Recall and F1 are macro averages, and AUC is the one-vs-rest AUC averaged over all classes.

### Comparison Table

Results on the held-out test data (`test_data.csv`, 2,723 rows):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9280 | 0.9953 | 0.9399 | 0.9399 | 0.9398 | 0.9130 |
| Decision Tree | 0.9049 | 0.9828 | 0.9194 | 0.9155 | 0.9174 | 0.8850 |
| KNN | 0.9269 | 0.9881 | 0.9402 | 0.9395 | 0.9398 | 0.9117 |
| Naive Bayes | 0.8957 | 0.9919 | 0.9052 | 0.9069 | 0.9057 | 0.8744 |
| Support Vector Machine | 0.9295 | 0.9953 | 0.9419 | 0.9407 | 0.9413 | 0.9147 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: 92.8% accuracy and MCC 0.913, only slightly behind SVM. Tuning picked C=100 (weak regularisation), and training accuracy (92.6%) is almost equal to test accuracy, so it is not overfitting. After scaling, most bean varieties can be separated with linear boundaries. |
| Decision Tree | Lowest scores apart from Naive Bayes (90.5% accuracy, MCC 0.885) and the lowest AUC (0.983), because a tree gives coarse leaf-based probabilities. Tuning chose max_depth=10 with min_samples_leaf=10. There is still a gap between training (94.0%) and test (90.5%) accuracy. It is the only model that misclassified some BOMBAY beans (96.2% recall), and its most common error was SIRA vs DERMASON (109 test beans). |
| KNN | 92.7% accuracy and F1 of 0.940, practically tied with Logistic Regression. Tuning picked k=7 with distance weighting. Scaling is essential because KNN uses distances. Its AUC (0.988) is lower than LR and SVM because probabilities come from only 7 neighbours. The model file is the largest (about 1.5 MB) because KNN stores the whole training set. |
| Naive Bayes | Still the weakest model (89.6% accuracy, MCC 0.874), because many features are strongly correlated (for example Area, ConvexArea and EquivDiameter), which breaks the independence assumption. The grid search chose a PowerTransformer before GaussianNB, which made the skewed features closer to Gaussian and raised test accuracy from 75.9% (plain GaussianNB) to 89.6%. Its AUC (0.992) is high, so it ranks classes well even when its top prediction is wrong. BARBUNYA was its hardest class (83.4% recall), and most of those errors were predicted as CALI (33 of 44). |
| Support Vector Machine | Best model, with the highest value in all six metrics (accuracy 92.9%, MCC 0.915; its AUC of 0.9953 is level with Logistic Regression at four decimals) and the best cross-validation F1 (0.943). With C=10 the RBF kernel fits slightly non-linear boundaries that Logistic Regression cannot. It is the slowest model to train because of the kernel and probability calibration. |
| Overall, Winner for your dataset? | SVM is the winner for this dataset. Logistic Regression and KNN are very close (MCC within 0.003). Logistic Regression would be a good simpler alternative because it is faster and easier to interpret. |

In every model the most common mistake was SIRA vs DERMASON, followed by BARBUNYA vs CALI or DERMASON vs SEKER. These pairs have similar size and shape. BOMBAY beans are much larger than the others and were classified perfectly by Logistic Regression, KNN and SVM.

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
python train_models.py   # tunes and saves all five models (about 15 seconds)
streamlit run app.py
```
