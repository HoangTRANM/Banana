# Business Use Case: Bananas' Quality Assessment for Market Allocation

## Contributoers: Inès Simond-Charbinat, Philomène Cholet, TRAN Ngoc Minh Hoang
  

## Business Challenge
1.1. Banana Market Overview

- Bananas are cultivated in over 150 countries, with a global production of approximately 105 million tonnes annually.
- Bananas are a staple fruit for both local consumption and international export, making quality consistency critical for market reputation and profitability.

1.2. Banana Production Issues

Over the past year, in our dataset, only 50% of harvested bananas could be sold at full price because the remaining half failed to meet quality standards: "bad" quality. Furthermore, global banana production faces major challenges:
- Panama Disease TR4 causes soil contamination: it delays the farming process of the fruit, already at a minimum of 9 months (soil preparation to harvest).
- Volatile freight rates and container shortages: it changes the profit each farmer and stakeholder will get from sales, so if the bananas are of bad quality, the business could be endangered.
- Strict European pesticide-residue limits: while most of the bananas are produced in East Asia, the South of Africa, and South America, farmers don't have the same base regulations, but the arriving produce in the EU still needs to meet certain standards.
- Rising labor activism in banana farms: it increases production and compliance costs while diminishing profit returns through less output of fruits.
- The long supply chain: 9-month growth, harvesting, refrigerated transport (2-3 weeks), controlled ripening, and stores dispatch means quality must be preserved through each stage, and especially early, otherwise losses occur downstream.

1.3. Problem Statement

Banana farmers need a reliable, systematic method to assess the quality of harvested bananas. The goal is to:
- Identify premium-quality ("good") batches for high-paying clients.
- Segregate lower-quality ("bad") batches for alternative markets.

1.4. Stakeholders

Primary Actor: Banana Farmers
Stakeholders
- Clients: supermarkets, distributors... seeking consistent quality.
- Logistics Partners: shipping companies, import (e.g., port) authorities... ensuring timely delivery.
- Regulators: food safety and pesticide authorities ensuring compliance.
- Local Workers & Plantation Operators: responsible for harvesting, handling, quality checking, and packing.
  
## 2. Dataset Description

The dataset used is banana_quality.csv, from Kaggle (https://www.kaggle.com/datasets/l3llff/banana?resource=download), containing 8,000 rows and 8 columns.
The dataset contains characteristics of bananas (size, weight, sweetness, softness, harvest time, ripeness, acidity, and quality), which are already scaled ("the data has been scaled, with a mean of 0, data_scaled = (data - data_mean) / data_std").

Main feature groups include:
- Size - size of fruit (a float between -10 and 10)
- Weight - weight of fruit (a float between -10 and 10)
- Sweetness - sweetness of fruit (a float between -10 and 10)
- Softness - softness of fruit (a float between -10 and 10)
- HarvestTime - amount of time passed from harvesting of the fruit (a float between -10 and 10)
- Ripeness - ripeness of fruit (a float between -10 and 10)
- Acidity - acidity of fruit (a float between -10 and 10)
- Quality - quality of fruit (a boolean: True for "Good" and False for "Bad")

## 3. Reproducibility — How to Run
**Step 1 — Import the Dataset**
In the Notebook, you should change how you import the dataset:
- If you have the CSV file in your Drive, we import our personal drive to use the CSV document.
```
from google.colab import drive
drive.mount('/content/drive')
```
- And we read the main CSV file of our analysis and name it. In this CSV file, the separator is a comma ','.
```
bq = pd.read_csv('/content/drive/MyDrive/csv/banana_quality.csv', sep=',')
```
- If the file is not in your Drive, just copy the path of the file "HERE":
```
bq = pd.read_csv('HERE', sep=',')
```

**Step 2 — Create and Activate a Virtual Environment**
Don't forget to install a Python environment:
```
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**Step 3 — Install Dependencies**
Here are the minimum required packages to pip install:
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Missingno
- Scikit-learn

**Step 4 — Run the full Notebook**

## 4. Baseline - Model Training
4.1. Features used
Training: all 7 numerical continuous columns; size, weight, sweetness, softness, harvesttime, ripeness, and acidity
Target: quality (boolean)

4.2. Preprocessing
The dataset is already Z-score scaled
Converted quality "Good"/"Bad" to boolean: True/False
Dropped duplicates
Train/test split = test size of 20% and a random state of 42
X as the training columns   
y as the target column


4.3. Baseline Model: 
- Linear SVC
- KNeighbors Classifier
- SVC
- Ensemble (Voting Classifier)

4.4. Baseline Metrics
- Accuracy scores
- Classification report:
  - Precision
  - Recall
  - F1-score
  - Support
- Conufsion matrix => the one we will look at the most

RESULTS OF GRID SEEARCHH!!!!!!

**Best LinearSVC parameters : {'C': 1, 'max_iter': 2000, 'random_state': 42}**

Test Accuracy: 0.879375

Confusion matrix:
 [[676 105]
 [ 88 731]]

Classification report:
               precision    recall  f1-score   support

       False       0.88      0.87      0.88       781
        True       0.87      0.89      0.88       819

    accuracy                           0.88      1600
   macro avg       0.88      0.88      0.88      1600
weighted avg       0.88      0.88      0.88      1600

**Best KNC params: {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'uniform'}**

KNC Accuracy: 0.97875

Confusion matrix:
 [[762  19]
 [ 15 804]]

Classification report:
               precision    recall  f1-score   support

       False       0.98      0.98      0.98       781
        True       0.98      0.98      0.98       819

    accuracy                           0.98      1600
   macro avg       0.98      0.98      0.98      1600
weighted avg       0.98      0.98      0.98      1600


**Best SVC params: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}**

SVC Accuracy: 0.984375

Confusion matrix:
 [[767  14]
 [ 11 808]]

Classification report:
               precision    recall  f1-score   support

       False       0.99      0.98      0.98       781
        True       0.98      0.99      0.98       819

    accuracy                           0.98      1600
   macro avg       0.98      0.98      0.98      1600
weighted avg       0.98      0.98      0.98      1600

## 5. Experiment Tracking

- Linear SVC
Change: we removed unnecessary parameters.
Impact: the accuracy score is unchanged at 0.879375
Conclusion: the model is not accurate enough; it may be because of underfitting. But we got a bad confusion matrix with 105 false positives and 88 false negatives.
-> In order to find the best performance, we perform a Grid Search to automatically test several parameter combinations:
   Best LinearSVC parameters : {'C': 1, 'max_iter': 2000, 'random_state': 42}
   Test Accuracy: 0.879375

- KNeighbors Classifier
Change: we tested n_neighbors = 3,5,7,9 (the main 4 ones used) and weights = uniform/distance.
Impact: the accuracy changed from around 0.978 to 0.981, with minor changes.
Conclusion: the best pair up was with a n_neighbors = 7, the distance didn't matter much in this case. The accuracy score is high, meaning overfitting due to a few rows. In terms of confusion matrix, we got between 17-19 false positives and 14-15 false negatives.

- SVC
Change: we simplified the hyperparameters, we kept C = 1.0, gamma = 'scale'.
Impact: the accuracy is unchanged as well at 0.98125.
Conclusion: we got a high accuracy score, meaning overfitting, because of the small number of rows in the dataset. But we got a good confusion matrix.
-> following exactly the same methodology than for the models above, we used GridSearchCV for optimizing the hyperparameters:
   Best SVC params: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
   SVC Accuracy: 0.984375

- Ensemble (Voting Classifier)
Change: we did an ensemble of LinearSVC, KNeighbors Classifier, and SVC with the best parameters found above.
Impact: the accuracy is high at 0.979375, and the confusion matrix looks reliable enough, both are the same as for the SVC ones.
Conclusion: even if the model is overfitting, SVC has the best confusion matrix results (only 14 false positives and 11 false negatives).

**CONCLUSION :**

The dataframe confirms that all models perform very well at predicting banana quality.

LinearSVC achieves about 88% accuracy, making it the weakest of the four
KNeighbors Classifier workd also very very well with nearly 98% accuracy
SVC performs the best with 98.4% accuracy and the highest F1 score, showing it balances precision and recall most effectively
The Ensemble Classifier combines all three models and achieves 97.9% accuracy
--> The model we choose is SVC, because it has the best accuracy and the best TP/TF results.

Remark : choosing a model with 98% accuracy is a lot. It very likely overfits, since we don't have a lot of rows on the dataset, so the models can memorize the data.
