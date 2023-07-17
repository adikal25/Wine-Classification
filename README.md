# Wine-Classification
Overview

This repository contains a Python script for building a K-Nearest Neighbors (KNN) model to classify wines based on a dataset from the scikit-learn library. The dataset consists of various attributes of different wines, and the KNN model is trained to predict the wine's class or type.

Dataset

The dataset used for this project is the Wine dataset available in the scikit-learn library. It contains attributes of three different types of wines. Each data sample has 13 features, including measurements of chemical properties, such as alcohol content, total phenols, flavonoids, etc. The dataset is loaded and used to train the KNN model.

Requirements

Python 3.x
scikit-learn
pandas
matplotlib
You can install the required libraries using pip:

Copy code
pip install scikit-learn pandas matplotlib
Usage

Clone the repository or download the code files.
Ensure you have installed the required libraries mentioned in the "Requirements" section.
Run the wine_classification.py script:
Copy code
python main.py
Model Details

The KNN model is used for classifying wines into one of the three classes. To achieve this, the dataset is split into training and testing sets. The model is trained using 80% of the data and then evaluated on the remaining 20% to calculate the accuracy.

Results

The trained KNN model achieved an accuracy of 80.55% on the test dataset. The accuracy represents the proportion of correct predictions out of the total predictions made by the model.

Conclusion

This project demonstrates how to use the K-Nearest Neighbors algorithm to classify wines based on their attributes. With an accuracy of 80.55%, the model shows promising results in predicting the wine's class. You can further experiment with hyperparameter tuning, feature engineering, or try other classification algorithms to improve the accuracy.

Feel free to edit and improve the accuracy of the code  and make changes to the dataset to get more accuracy

