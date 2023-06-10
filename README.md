# Fake_News_Classifie

This project aims to build a fake news classifier using Natural Language Processing. The classifier is trained on a dataset containing news headlines and corresponding information.

## Getting Started

To get started with the project, you need to have the following libraries installed:

-   `nlp_utils`
-   `pandas`
-   `seaborn`
-   `sklearn`

You can install these libraries using the package manager `pip`.

## Loading the Data

The dataset used for training the classifier is stored in a CSV file named `train.csv`. 

The loaded dataset contains 20800 rows and 5 columns.

## Data Preprocessing

Before training the classifier, some preprocessing steps are performed on the text data. These steps include:

-   Removing numbers attached to letters
-   Converting all strings to lowercase
-   Removing punctuation
-   Replacing newline characters with spaces
-   Removing non-ASCII characters
-   Removing stop words and stemming the text

## Splitting the Data

The dataset is split into training and testing sets with a 70% - 30% ratio. The training set is used to train the classifier, and the testing set is used to evaluate its performance.

## Feature Extraction

Two different feature extraction techniques are used: TF-IDF vectorization and Count vectorization.

-   TF-IDF Vectorization: The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is applied to the training and testing data using the TfidfVectorizer class from `sklearn.feature_extraction.text`. This process tokenizes the documents, learns the vocabulary, and encodes the documents based on their TF-IDF weights.
    
-   Count Vectorization: The CountVectorizer class from `sklearn.feature_extraction.text` is used to tokenize the documents and build a vocabulary. It encodes the documents based on the count of each word in the vocabulary.
    

## Machine Learning

Two machine learning models are trained and evaluated:

1.  Naive Bayes Model:
    
    -   The Naive Bayes model is trained on the TF-IDF vectorized data and the Count vectorized data.
    -   The accuracy and confusion matrix are calculated for both cases.
    -   The wrong predictions are displayed along with the total predictions made.
    ![download](https://github.com/pras-ops/Fake_News_Classifier/assets/56476064/9d3cf4e7-9230-42af-b3bc-e72cd60008da)

2.  Random Forest Model:
    
    -   The Random Forest model is trained on the TF-IDF vectorized data and the Count vectorized data.
    -   The accuracy and confusion matrix are calculated for both cases.
    -   The wrong predictions are displayed along with the total predictions made.
    -   |  Model accuracy on train  |0.9999218688960075 |
    -   | Model accuracy on test     |0.9050309879693766|
3.  K-Nearest Neighbors Model:
    
    -   The K-Nearest Neighbors model is trained on the TF-IDF vectorized data and the Count vectorized data.
    -   The accuracy and confusion matrix are calculated for both cases.
    -   The wrong predictions are displayed along with the total predictions made.
    -   |  Model accuracy on train  | 0.5187123994062036 |
    -   | Model accuracy on test     | 0.48651111921254103|
