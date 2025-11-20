# Business Use Case: Banana Quality Assessment for Market Allocation

## Contributoers: Ilnes, Philomène, TRAN Ngoc Minh Hoang
  

## Business Challenge
1.1. Banana Market overview

- Bananas are cultivated in over 150 countries, with a global production of approximately 105 million tonnes annually.
- Bananas are a staple fruit for both local consumption and international export, making quality consistency critical for market reputation and profitability.

1.2. Banana production issues

Over the past year, only 50% of harvested bananas could be sold at full price because the remaining half failed to meet quality standards. Global banana production faces major challenges:
- Panama Disease TR4 causing soil contamination
- Volatile freight rates and container shortages
- Strict European pesticide-residue limits
- Rising labor activism increasing production and compliance costs
- The long supply chain (9-month growth, green harvesting, refrigerated transport, controlled ripening) means quality must be preserved early, otherwise losses occur downstream.

1.3. Problem Statement

Banana farmers need a reliable, systematic method to assess the quality of harvested bananas. The goal is to:
- Identify premium-quality batches for high-paying clients.
- Segregate lower-quality batches for alternative markets.

1.4. Stakeholders

Primary Actor: Banana Farmers/ Quality Inspectors
Stakeholders
- Export Clients: Supermarkets, distributors seeking consistent quality.
- Logistics Partners: Shipping companies, port authorities ensuring timely delivery.
- Regulators: Food safety and pesticide authorities ensuring compliance.
- Local Workers & Plantation Operators: Responsible for harvesting, handling, and packing.
  
## 2. Dataset Description

The dataset used is banana_quality.csv, containing 8,000 rows and 8 columns.
Dataset contains characteristics of bananas (size, weight, sweetness, softness, harvest time, ripeness, acidity and quality)

- Main feature groups include:
Size - size of fruit
Weight - weight of fruit
Sweetness - sweetness of fruit
Softness - softness of fruit
HarvestTime - amount of time passed from harvesting of the fruit
Ripeness - ripness of fruit
Acidity - acidity of fruit
Quality - quality of fruit

## 3. Reproducibility — How to Run From a Fresh Clone
**Precise instructions to reproduce your results: where to copy the dataset file, python version, how to install dependencies, how to run the training pipeline, and anything else needed from a fresh copy of the repository**

**Step 1 — import the Dataset**

- We import our personal drive to use the CSV document.
from google.colab import drive
drive.mount('/content/drive')
- We read the main CSV file of our analysis and name it.
bq = pd.read_csv('/content/drive/MyDrive/csv/banana_quality.csv', sep=',')

Or from kaggle: https://www.kaggle.com/datasets/l3llff/banana/code

Install dependencies as needed:
pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

Set the path to the file you'd like to load
file_path = ""

Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "l3llff/banana",
  file_path,
See the documenation for more information:
https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas


**Step 2 — Python Version: Python 3.10+**
**Step 3 — Create and Activate Virtual Environment**
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

**Step 4 — Install Dependencies**

Minimum required packages:

import numpy as np              # For ca
import pandas as pd             # For functions
import seaborn as sns           # For display formatting
import matplotlib.pyplot as plt # For creating histograms, plots, and more
import missingno                # For visualizing missing values

**Step 5 — Run the Notebook: v4_bcu_eda_ml_bananas_quality.ipynb**

## 4. The baseline used: which features, which pre-processing, which model and the metrics obtained.

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = bq[cols]            # already without the column "quality"
y = bq['quality']

4.1. Features used
All 7 numerical columns: size, weight, sweetness, softness, harvesttime, ripeness, acidity, Target, quality (boolean)

4.2. Preprocessing
Dataset is already Z-score scaled
Converted quality "Good"/"Bad" → True/False
Dropped duplicates
Train/test split = 80 / 20

4.3. Baseline Model: 
- LinearSVC
- KNeighborsClassifier
- SVC

4.4. Baseline Metrics

                   model  trial_number  accuracy_score:  \
0              LinearSVC             1         0.879375   
1              LinearSVC             2         0.879375   
2   KNeighborsClassifier             1         0.980625   
3   KNeighborsClassifier             2         0.980625   
4   KNeighborsClassifier             3         0.979375   
5   KNeighborsClassifier             4         0.979375   
6   KNeighborsClassifier             5         0.980000   
7   KNeighborsClassifier             6         0.978750   
8   KNeighborsClassifier             7         0.980000   
9   KNeighborsClassifier             8         0.980000   
10                   SVC             1         0.981250   
11                   SVC             2         0.981250   
12   ensemble classifier             2         0.981250   

          confusion_matrix:  
0   [[676, 105], [88, 731]]  
1   [[676, 105], [88, 731]]  
2    [[764, 17], [14, 805]]  
3    [[764, 17], [14, 805]]  
4    [[762, 19], [14, 805]]  
5    [[762, 19], [14, 805]]  
6    [[764, 17], [15, 804]]  
7    [[762, 19], [15, 804]]  
8    [[763, 18], [14, 805]]  
9    [[763, 18], [14, 805]]  
10   [[766, 15], [15, 804]]  
11   [[766, 15], [15, 804]]  
12   [[766, 15], [15, 804]]  

## 5. Experiment tracking: what did you change, why and how it impacted the metrics.

- LinearSVC
Change: Removed unnecessary parameters.
Impact: Accuracy unchanged at 0.879. Underfits, linear boundary insufficient.

- KNeighborsClassifier
Change: Tested n_neighbors = 3,5,7,9 and weights = uniform/distance.
Impact: Accuracy ~0.978–0.981, minor changes; best with k=5. Stable and strong.

- SVC (RBF)
Change: Simplified hyperparameters (C=1.0, gamma='scale').
Impact: Accuracy 0.98125, best individual model.

- Ensemble (Voting)
Change: Combined LinearSVC + KNN + SVC.

Impact: Accuracy same as SVC (0.98125)
choose model 98% is a lot because overfitting since not a lo rows / choose svc because of best results and best TP TF




