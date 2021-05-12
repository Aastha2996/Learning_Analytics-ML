
# 1. import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report
import process_file
import feature_cal as features


# 2. Load data
def load_data():
    """
    Load Data function

        df -> Loaded dasa as Pandas DataFrame
    """
    plagiarism_df = pd.read_csv('data/students/student_info.csv')

    return plagiarism_df 


def split_train_test_set(merge_df):
    # Remove student_6's answer sheet as we are considering as a base answer sheet.
    merge_df_drop = merge_df.drop([5], axis=0)
    Copy_X = merge_df_drop.iloc[:, [4,14,24]] 
    Copy_y = merge_df_drop.iloc[:, 2]
    print(merge_df)
    return model_selection.train_test_split(Copy_X, Copy_y, test_size=0.26, random_state=3)

def train_naive_model_classifier(Train_X, Train_Y):
    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    return Naive.fit(Train_X, Train_Y)

def evaluate_model(model, Test_X, Test_Y):
    # predict the labels on validation dataset
    predictions_NB = model.predict(Test_X)
    print("Test X")
    print(Test_X)
    # Use accuracy_score function to get the accuracy
    print('\nPredicted class labels: ')
    print(predictions_NB)
    print('\nTrue class labels: ')
    print(Test_Y.values)

    print("\nNaive Bayes Accuracy Score -> ",accuracy_score(Test_Y, predictions_NB)*100)
    print(classification_report(Test_Y,predictions_NB, zero_division=0))
    pass

def save_model(model):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
    
    """

    filename = 'models/classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))
    pass

def main():
    """
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Load CSV File
        2) Data cleaning and pre-processing
    """
    
    print('Load Data')
    print('------------------------')

    plagiarism_df = load_data()

    print('process file and merge text file in existing dataframe')
    print('------------------------')
    text_df = process_file.create_text_column(plagiarism_df)
    print('Feature Engineering steps')
    print('------------------------')

    print('containment value for student_5.txt file compaired  with student_6.txt (base file)', features.calculate_containment(text_df, 1, 'student_5.txt'))
    
    lcs = features.lcs_norm_word(text_df.iloc[4]['Text'], text_df.iloc[5]['Text']) 
    print('LCS value for for student_5.txt file compaired  with student_6.txt (base file) ', lcs)

    print('Create All Features')
    print('------------------------')

    features_df = features.cal_all_features(text_df)
    print(features_df)

    print("Merge features into existing dataframe")
    merge_df = pd.merge(text_df, features_df, right_index=True, left_index=True)

    print("split train and test dataset")
    print('------------------------')
    Train_X, Test_X, Train_Y, Test_Y = split_train_test_set(merge_df)
    # print(Train_X, Train_Y)
    # merge_df_drop = merge_df.drop([5], axis=0)
    # # print(merge_df_drop.iloc[:, [0,5,8,10]] )
    # Copy_X = merge_df_drop.iloc[:, [5,8,10]] 
    # Copy_y = merge_df_drop.iloc[:, 2]

    print('train model using Naive Bayes Classifier')
    print('------------------------')
    model = train_naive_model_classifier(Train_X, Train_Y)
    # print(Copy_X)
    # print(Copy_y)
    print('evaluate model')
    print('------------------------')

    evaluate_model(model, Test_X, Test_Y)

    print('save model')
    print('------------------------')
    save_model(model)


if __name__ == '__main__':
    main()