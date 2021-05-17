import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
sc = StandardScaler()

def load_data():
    """
    Load Data function

        df -> Loaded dasa as Pandas DataFrame
    """
    df = pd.read_csv('data/dropout/StudentsPerformance.csv')
    return df

def split_train_test_set(X, y):
    """
    Split Training and Test Dataset function

    """
    return train_test_split(X, y, test_size = 0.2, random_state = 0)

def feature_scaling(X_train, X_test):
    """
    Standardization Method

    """
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def train_random_forest_classifier(X_train, y_train):
    # fit the training dataset on the Random Forest classifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    return classifier.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    # predict the labels on validation dataset
    y_pred = model.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix :", cm)
    print("Accuracy Score is :", accuracy_score(y_test, y_pred))
    pass

def save_model(model):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
    
    """

    filename = 'models/dropout_classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))
    pass

def save_X_train(X_train):
    """
    Standardization Method

    """
    filename = 'models/dropout_X_train.pkl'
    pickle.dump(X_train, open(filename, 'wb'))
    pass

""" #def save_visualisation(model, X_test, y_test):
    X_set, y_set = sc.inverse_transform(X_test), y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                        np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 0.25))
    plt.contourf(X1, X2, model.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                alpha = 0.7, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ['red', 'green'][i], label = j)
    plt.title('Random Forest Classification (Test set)')
    plt.xlabel('First Exam')
    plt.ylabel('Second Exam')
    plt.legend()
    plt.show()
    plt.savefig('app/static/plot.png')
    plt.close()
    pass """

def main():
    """
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Load CSV File
        2) Data cleaning and pre-processing
    """
    
    print('Load Data')
    print('------------------------')

    df = load_data()

    print('split target and feature values')
    print('------------------------')
    X = df.iloc[:, [0,1,2,4,5,6,7,8]].values
    y = df.iloc[:, -1].values
    print(X, y)

    print('Encoding categorical data')
    print('------------------------')
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    print(X)

    print('Split Train and Test Dataset')
    print('------------------------')
    X_train, X_test, y_train, y_test =  split_train_test_set(X, y)
    print(X_train)

    print('Standardized Training Dataset')
    print('------------------------')
    X_train_sc, X_test_sc = feature_scaling(X_train, X_test)
    print(X_train_sc)

    print("split train and test feature values")
    print('------------------------')
    X_train_sc, X_test_sc = feature_scaling(X_train, X_test)

    print('save X train data')
    print('------------------------')
    save_X_train(X_train)

    print('train model using Random Forest Classifier')
    print('------------------------')
    model = train_random_forest_classifier(X_train_sc, y_train)
    
    print('evaluate model')
    print('------------------------')

    print(model.predict(sc.transform([[1,0,1,0,0,0,0,1,0,0,0,0,55,10,46,49,39]])))

    evaluate_model(model, X_test_sc, y_test)

    print('save model')
    print('------------------------')
    save_model(model)

   # print('save visualisation')
    #print('------------------------')
    #save_visualisation(model, X_test_sc, y_test)


if __name__ == '__main__':
    main()