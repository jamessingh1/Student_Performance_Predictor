# student_performance/app.py
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(_name_)


model = joblib.load('model/model.pkl')


UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

default_data_path = 'student_data.csv'
default_df = pd.read_csv(default_data_path)

X = default_df[['Hours_Studied', 'Attendance', 'Previous_Grades']]
default_df['Predicted_Score'] = model.predict(X)

plt.figure(figsize=(8, 5))
plt.hist(default_df['Predicted_Score'], bins=10, color='skyblue', edgecolor='black')
plt.title('Predicted Scores Distribution')
plt.xlabel('Score')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_FOLDER, 'score_plot.png'))  # Save to static folder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])
    previous = float(request.form['previous'])

    input_data = pd.DataFrame([[hours, attendance, previous]],
                              columns=['Hours_Studied', 'Attendance', 'Previous_Grades'])
    prediction = model.predict(input_data)[0]
    return render_template('result.html', prediction=round(prediction, 2))

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['csv_file']
    if file.filename == '':
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    if {'Hours_Studied', 'Attendance', 'Previous_Grades'}.issubset(df.columns):
        X = df[['Hours_Studied', 'Attendance', 'Previous_Grades']]
        df['Predicted_Score'] = model.predict(X)

        plt.figure(figsize=(8, 5))
        plt.hist(df['Predicted_Score'], bins=10, color='skyblue', edgecolor='black')
        plt.title('Predicted Scores Distribution')
        plt.xlabel('Score')
        plt.ylabel('Number of Students')
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_FOLDER, 'score_plot.png'))

        return render_template('batch_result.html',
                               tables=[df.to_html(classes='table table-striped', index=False)],
                               image='score_plot.png')
    else:
        return "CSV must contain 'Hours_Studied', 'Attendance', and 'Previous_Grades' columns."


if _name_ == '_main_':
    app.run(debug=True)
