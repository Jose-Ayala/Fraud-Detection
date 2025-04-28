# Fraud Detection with Data Imbalance

This project investigates credit card fraud detection with a focus on addressing the challenges posed by class imbalance. I analyzed two distinct datasets – the Credit Card Fraud Detection dataset from Kaggle and a Simulated Credit Card Transactions dataset generated using Sparkov. I applied different data balancing techniques, trained an MLP neural network model, evaluated its performance, and used SHAP analysis to explain the model's predictions.

## Project Overview

The primary goal of this project was to understand the significant impact of **class imbalance** on fraud detection model performance and to explore effective methods for mitigating this issue. Class imbalance occurs when the number of observations in one class (the minority class, e.g., fraudulent transactions) is significantly lower than the number of observations in the other class (the majority class, e.g., legitimate transactions). This imbalance is a common characteristic of fraud datasets and can severely hinder a model's ability to correctly identify the rare, but crucial, fraudulent instances.

I worked with two publicly available credit card transaction datasets to analyze how varying degrees of imbalance affect model training and evaluation. The project involved comprehensive data preprocessing, training an MLP model on both imbalanced and balanced data, evaluating performance using relevant metrics (AUC, Precision, Recall), and interpreting the model's decisions using SHAP for explainability.

## Files

* `Fraud Detection with Data Imbalance.pdf`: The project report detailing the methodology, findings, and conclusions.
* `FraudDetection.ipynb`: The main Jupyter Notebook containing the code for data loading, preprocessing, model training, evaluation, and SHAP analysis.
* `requirements.txt`: A list of necessary Python libraries to run the project.
* `README.md`: This file, providing an overview of the project.

## Datasets

The datasets used in this project are not included directly in this repository due to their size, but can be downloaded from the following sources:

### Credit Card Fraud Detection

* **Source URL:** https://www.kaggle.com/mlg-ulb/creditcardfraud/
* **Source License:** https://opendatacommons.org/licenses/dbcl/1-0/
* **Variables:** PCA transformed features (V1-V28), Time, Amount. The dataset is highly imbalanced with a very small percentage of fraudulent transactions.
* **Fraud Category:** Card Not Present Transaction Fraud
* **Provider:** Machine Learning Group - ULB
* **Release Date:** 2018-03-23
* **Description:** This dataset contains anonymized credit card transactions by European cardholders in September 2013. It includes 492 frauds out of 284,807 transactions over 2 days. The numerical features are the result of a PCA transformation, with non-transformed Time and Amount features also included.

### Simulated Credit Card Transactions generated using Sparkov

* **Source URL:** https://www.kaggle.com/kartik2112/fraud-detection
* **Source License:** https://creativecommons.org/publicdomain/zero/1.0/
* **Variables:** Transaction date, credit card number, merchant, category, amount, name, street, gender. All variables are synthetically generated.
* **Fraud Category:** Card Not Present Transaction Fraud
* **Provider:** Kartik Shenoy
* **Release Date:** 2020-08-05
* **Description:** This is a simulated credit card transaction dataset generated using the Sparkov Data Generation tool, based on a version modified for Kaggle. It covers transactions of 1000 customers with a pool of 800 merchants over 6 months. Both train and test segments were used directly from the source, with the test segment randomly downsampled.

## Dependencies

The key Python libraries required to run the `FraudDetection.ipynb` notebook are:

* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations.
* `scikit-learn`: For splitting data, preprocessing (like scaling), and potentially other utilities.
* `tensorflow`: For building and training the neural network model using Keras API.
* `imblearn`: For handling class imbalance using techniques like RandomOverSampler and RandomUnderSampler.
* `shap`: For model explainability analysis.
* `matplotlib`: For creating visualizations.

These dependencies are listed in the `requirements.txt` file for easy installation.

## Setup and Installation

1.  Clone the repository:

    ```bash
    git clone [repository_url]
    ```

2.  Navigate to the project directory:

    ```bash
    cd [repository_name]
    ```

3.  Install the required Python packages. It's highly recommended to use a virtual environment:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the datasets:** Obtain the `creditcard.csv` file from the Kaggle Credit Card Fraud Detection source URL and the necessary file(s) (likely CSV) from the Kaggle Simulated Credit Card Transactions source URL.
5.  **Place the downloaded data files in the same directory as the Jupyter Notebook file (`FraudDetection.ipynb`).**

## Usage

1.  Ensure you have followed the setup steps and downloaded the necessary datasets, placing them in the same directory as `FraudDetection.ipynb`.
2.  Open the `FraudDetection.ipynb` Jupyter Notebook using a compatible environment (like JupyterLab or VS Code with the Jupyter extension).
3.  Run the cells sequentially from top to bottom. The notebook contains all the code for data loading, preprocessing, applying balancing techniques, training the MLP models, evaluating their performance, and conducting the SHAP analysis.
4.  The output of the cells will display performance metrics and generate visualizations, including SHAP plots.

## Methodology

1.  **Data Loading and Preprocessing:**
    * Loaded the Credit Card Fraud Detection dataset and the Simulated Credit Card Transactions dataset.
    * For the Simulated data, I applied one-hot encoding to categorical features and scaling to numerical features.
    * Split the Credit Card dataset into stratified training and validation sets (80/20) to maintain the class distribution.
    * Applied oversampling (e.g., using SMOTE) and undersampling techniques to the training data of both datasets to create balanced versions for comparative analysis.

2.  **Model Architecture:**
    * Employed a simple Multilayer Perceptron (MLP) neural network architecture consisting of an input layer, two hidden layers with 64 and 32 neurons (using ReLU activation), and a single output neuron with a sigmoid activation function for binary classification.

3.  **Model Training:**
    * Trained the MLP model separately on the original imbalanced training data, the oversampled training data, and the undersampled training data for each dataset.
    * Training was performed using the Adam optimizer, binary cross-entropy loss, a batch size of 32, and for 10 epochs.

4.  **Evaluation:**
    * Evaluated the performance of each trained model on the **original, untouched validation sets** using key metrics appropriate for imbalanced data: AUC (Area Under the ROC Curve), Accuracy, Precision, and Recall.

5.  **Explainability:**
    * Utilized SHAP (SHapley Additive exPlanations) analysis, particularly focusing on the model trained on the oversampled Credit Card data, to understand which features were most influential in the model's predictions and how individual feature values impacted the decision for specific transactions.

## Results

* **Impact of Imbalance (Credit Card Data):** The model trained on the original imbalanced data showed poor fraud detection capability (AUC ~0.5, precision/recall near zero), highlighting a strong bias towards the majority class despite high overall accuracy.
* **Performance After Balancing (Credit Card Data):**
    * **Oversampling:** Led to a significant improvement in fraud detection, achieving a high AUC (~0.95) and substantially increased recall (~0.89) on the original validation set, albeit with lower precision (~0.04).
    * **Undersampling:** Showed a modest improvement in AUC (~0.56) compared to the imbalanced data, with a precision of ~0.50 and a recall of ~0.13 on the validation set.
* Addressing class imbalance through techniques like oversampling was crucial for enabling the model to effectively identify the rare fraudulent transactions, demonstrating clear trade-offs between metrics depending on the balancing strategy.
* **SHAP Insights:** SHAP analysis confirmed the importance of features like `Time` and `Amount` in driving the model's predictions for the Credit Card dataset. Individual SHAP decision plots provided granular insight into how the values of key features pushed predictions towards or away from the fraudulent class.

## Report

A detailed account of the project, including the full methodology, comprehensive results, challenges encountered, and further discussion, is available in the report file: `Fraud Detection with Data Imbalance.docx`.

## Author

Jose Ayala
