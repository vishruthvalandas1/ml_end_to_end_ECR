from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Validate form inputs
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('ethnicity')  
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score_str = request.form.get('reading_score')  
        writing_score_str = request.form.get('writing_score')  
        
        # Validate required fields
        if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score_str, writing_score_str]):
            return render_template('home.html', results="Error: All fields are required")
        
        # Type assertions after validation
        assert gender is not None
        assert race_ethnicity is not None
        assert parental_level_of_education is not None
        assert lunch is not None
        assert test_preparation_course is not None
        assert reading_score_str is not None
        assert writing_score_str is not None
        
        try:
            reading_score = float(reading_score_str)
            writing_score = float(writing_score_str)
        except ValueError:
            return render_template('home.html', results="Error: Invalid score values")
        
        data=CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            writing_score=writing_score,
            reading_score=reading_score
        )
        try:
            pred_df=data.get_data_as_data_frame()
            print(pred_df)
            
            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(pred_df)
            return render_template('home.html',results=results[0])
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(error_msg)
            return render_template('home.html', results=error_msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)