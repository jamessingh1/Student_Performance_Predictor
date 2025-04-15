import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('student_data.csv')

X = df[['Hours_Studied', 'Attendance', 'Previous_Grades']]
y = df['Final_Score']
df.columns = df.columns.str.strip()  # Removes any leading/trailing spaces


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'model/model.pkl')
print("Model trained and saved!")
