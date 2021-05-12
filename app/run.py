from flask import Flask, render_template, request, jsonify, Response, url_for
import plotly
from plotly.graph_objs import Pie
import json
import joblib
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'models')

import feature_cal as features
import process_file
import os

# app = Flask(__name__, static_url_path = "/tmp", static_folder = "tmp")

app = Flask(__name__)

# load model
model = joblib.load("models/classifier.pkl")
dropout_model = joblib.load("models/dropout_classifier.pkl")
dropout_X_train = joblib.load("models/dropout_X_train.pkl")
sc = StandardScaler()

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    df = pd.read_csv('data/students/student_info.csv')
    plagiarism = df.iloc[:,2:].values
    is_plagiarised = 0
    for p in plagiarism:
        if p[0] == 1:
            is_plagiarised +=1
    
    plagiarised_per = round(is_plagiarised/(len(plagiarism)-1)*100)
    non_plagiarised_per = 100 - plagiarised_per
    # extract data needed for visuals
    graphs = [
            # GRAPH 1 - genre graph
        {
            'data': [
                Pie(
                    values=[plagiarised_per, non_plagiarised_per],
                    labels=['Plagiarised', 'Non-Plagiarised']
                )
            ],

            'layout': {
                'title': 'Distribution of Plagiarised or Non Plagiarised',
                'height': 700,
                'width': 800
            }
        }
    ]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# plagiarism detection webpage displays cool visuals and receives user input text for model
@app.route('/plagiarism-detection')
def plagiarism():
   
    # create visuals
    
    # render web page with plotly graphs
    return render_template('plagiarism_detection.html')

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    text = process_file.process_file(query, False)
    text_length = len(text.split(" "))
    print(text_length)
    print(query)
    if text_length < 22:
         return render_template(
            'go.html',
            query=query,
            is_plagiarised='not enough to detect plagiarism.'
        )
    text_df = pd.DataFrame ({'File': ['student_6.txt'], 'Task': ['a']}, columns = ['File', 'Task'])
    text_df = process_file.create_text_column(text_df)
    print(text_df)
    data = {
            'File': ['New_File.txt'],
            'Task': ['a'],
            'Text':  [text]
            }
    df = pd.DataFrame (data, columns = ['File','Task','Text'])
    text_df = text_df.append(df, ignore_index=True)
    print(text_df)
    features_df = features.cal_all_features(text_df)
    
    merge_df = pd.merge(text_df, features_df, right_index=True, left_index=True)
    merge_df_drop = merge_df.drop([0], axis=0)
    Copy_X = merge_df_drop.iloc[:, [3,13,23]] 
    Copy_y = merge_df_drop.iloc[:, 2]
    predictions_NB = model.predict(Copy_X)
    print(Copy_X)
    print(predictions_NB)
    check_plagiarism = predictions_NB[0]

    if check_plagiarism == 0:
        is_plagiarised = 'Not Plagiarised'
    else:
        is_plagiarised = 'Plagiarised'

    return render_template(
        'go.html',
        query=query,
        is_plagiarised=is_plagiarised
    )

# student-dropout webpage displays cool visuals and receives user input text for model
@app.route('/student-dropout')
def dropOut():
   return render_template('dropout.html')

# web page that handles user query and displays model results
@app.route('/dropout_go')
def dropout_go():
    # save user input
    gender = request.args.get('gender', '')
    group = request.args.get('group', '')
    level = request.args.get('level', '')
    attendance = request.args.get('attendance', '') 
    duration = request.args.get('duration', '')
    maths_score = request.args.get('maths_score', '') 
    reading_score = request.args.get('reading_score', '') 
    writing_score = request.args.get('writing_score', '') 

    # fit trained data
    sc.fit_transform(dropout_X_train)

    if gender=='male':
        if group=="group A":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,1,0,0,0,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,0,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,1,0,0,0,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[0,1,1,0,0,0,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[0,1,1,0,0,0,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[0,1,1,0,0,0,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))              

        elif group=="group B":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,1,0,0,0,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,1,0,0,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,1,0,0,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[0,1,0,1,0,0,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[0,1,0,1,0,0,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[0,1,0,1,0,0,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))
        
        elif group=="group C":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,1,0,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,1,0,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,1,0,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,1,0,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,1,0,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,1,0,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))
       
        elif group=="group D":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,1,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,1,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,1,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,1,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,1,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,1,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))
    
        else:
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,0,1,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,0,1,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,0,1,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,0,1,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,0,1,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[0,1,0,0,0,0,1,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))


    else:
        if group=="group A":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,1,0,0,0,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,0,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,1,0,0,0,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[1,0,1,0,0,0,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[1,0,1,0,0,0,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[1,0,1,0,0,0,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))              

        elif group=="group B":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,1,0,0,0,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,1,0,0,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,1,0,0,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[1,0,0,1,0,0,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[1,0,0,1,0,0,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[1,0,0,1,0,0,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))
        
        elif group=="group C":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,1,0,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,1,0,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,1,0,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,1,0,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,1,0,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,1,0,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))
       
        elif group=="group D":
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,1,0,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,1,0,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,1,0,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,1,0,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,1,0,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,1,0,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))
        else:
            if level=="Master's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,0,1,1,0,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Associate's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,0,1,0,1,0,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Bachelor's Degree":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,0,1,0,0,1,0,0,0,attendance, duration, maths_score, reading_score, writing_score]]))  
            elif level=="Some College":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,0,1,0,0,0,1,0,0,attendance, duration, maths_score, reading_score, writing_score]]))
            elif level=="Some High School":
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,0,1,0,0,0,0,1,0,attendance, duration, maths_score, reading_score, writing_score]]))
            else:
                dropout = dropout_model.predict(sc.transform([[1,0,0,0,0,0,1,0,0,0,0,0,1,attendance, duration, maths_score, reading_score, writing_score]]))

    print(dropout)
    check_dropout = dropout[0]
    print("heree ",check_dropout)
    if check_dropout == 1:
        message= 'There is a high chance that this student will drop out.'
    else:
        message = 'There is a high chance that this student will continue the program.'

    return render_template(
        'dropout_go.html',
        gender=gender,
        group=group,
        level=level,
        attendance= attendance,
        duration= duration,
        maths_score= maths_score,
        reading_score= reading_score,
        writing_score= writing_score,
        message = message
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()