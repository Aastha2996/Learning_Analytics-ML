# 1. import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def containment(ngram_array):
    ''' Containment is a measure of text similarity. It is the normalized, 
       intersection of ngram word counts in two texts.
       :param ngram_array: an array of ngram counts for an answer and source text.
       :return: a normalized containment value.'''
    # the intersection can be found by looking at the columns in the ngram array
    # this creates a list that holds the min value found in a column
    # so it will hold 0 if there are no matches, and 1+ for matching word(s)
    intersection_list = np.amin(ngram_array, axis=0)
    
    # optional debug: uncomment line below
    # print(intersection_list)

    # sum up number of the intersection counts
    intersection = np.sum(intersection_list)
    
    # count up the number of n-grams in the answer text
    answer_idx = 0
    answer_cnt = np.sum(ngram_array[answer_idx])
    
    # normalize and get final containment value
    containment_val =  intersection / answer_cnt
    
    return containment_val

# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    
    # compairing with the original student answer sheet
    source_filename = 'student_6.txt'
    a_text = df[df['File'] == answer_filename]['Text'].values[0]
    s_text = df[df['File'] == source_filename]['Text'].values[0]


    # instantiate an ngram counter
    counts = CountVectorizer(analyzer='word', ngram_range=(n,n))
    
    # create array of n-gram counts for the answer and source text
    ngrams = counts.fit_transform([a_text, s_text])
    ngram_array = ngrams.toarray()
    
    
    return containment(ngram_array)

# Compute the normalized LCS given an answer text and a source text
def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''
    
    # Splitting answer_text & source_text into list of words
    list_A = answer_text.split()
    word_count_A = len(list_A)
    list_S = source_text.split()
    
    # Create a matrix of zeroes with an additional row/column where rows is formed by length of source text words
    # and columns is formed by length of answer text words

    # we can say this as a vector matrix also and divide both sheet's sentences into words
    # create matrix from those words
    lcs_matrix = np.zeros((len(list_S) + 1, len(list_A) + 1), dtype=int)
    
    # Fill this matrix up by traversing the words in answer_text and source_text 
    for r_idx,r_word in enumerate(list_S, 1):
        for c_idx,c_word in enumerate(list_A, 1):
            if c_word == r_word:
                lcs_matrix[r_idx][c_idx] = lcs_matrix[r_idx-1][c_idx-1] + 1
            else:
                lcs_matrix[r_idx][c_idx] = max(lcs_matrix[r_idx][c_idx-1], lcs_matrix[r_idx-1][c_idx])
    
    # Longest Common Subsequence value
    lcs_val = lcs_matrix[len(list_S)][len(list_A)]
    
    # Normalized Longest Common Subsequence value
    lcs_norm = lcs_val/word_count_A
    
    return lcs_norm

# Function returns a list of containment features, calculated for a given n 
# Should return a list of length 100 for all files in a complete_df
def create_containment_features(df, n, column_name=None):
    print('Create Containment features!')

    containment_values = []
    
    if(column_name==None):
        column_name = 'c_'+str(n) # c_1, c_2, .. c_n
    
    # iterates through dataframe rows
    for i in df.index:
        
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i, 'File'] != 'student_6.txt':
          c = calculate_containment(df, n, file)
          containment_values.append(c)
        # Sets value to -1 for original tasks 
        else:
          containment_values.append(-1)

    print(str(n)+'-gram containment features created!')
    return containment_values

# Function creates lcs feature and add it to the dataframe
def create_lcs_features(df, column_name='lcs_word'):
    print('Create LCS features!')
    lcs_values = []
    
    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i, 'File'] != 'student_6.txt':
          answer_text = df.loc[i, 'Text'] 
          task = df.loc[i, 'Task']
          # we know that source texts have Class = -1
          orig_rows = df[(df['File'] == 'student_6.txt')]
          orig_row = orig_rows[(orig_rows['Task'] == task)]
          source_text = orig_row['Text'].values[0]

          # calculate lcs
          lcs = lcs_norm_word(answer_text, source_text)
          lcs_values.append(lcs)
        else:
          lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values

def cal_all_features(text_df):
    ngram_range = range(1,21)
    features_list = []

    # Create features in a features_df
    all_features = np.zeros((len(ngram_range)+1, len(text_df)))

    # Calculate features for containment for ngrams in range
    i=0
    for n in ngram_range:
        column_name = 'c_'+str(n)
        features_list.append(column_name)
        # create containment features
        all_features[i]=np.squeeze(create_containment_features(text_df, n))
        i+=1

    # Calculate features for LCS_Norm Words 
    features_list.append('lcs_word')
    all_features[i]= np.squeeze(create_lcs_features(text_df))
    # create a features dataframe
    features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

    # Print all features/columns
    print()
    print('Features: ', features_list)
    print()
    return features_df
