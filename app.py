import pandas as pd
import numpy as np
from flask import Flask, request, jsonify,url_for,render_template
import pickle
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('telco-customer-churn.csv')

# Preprocess the dataset
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].replace({'Yes': 1, 'No': 0})
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the models
lr = LogisticRegression()
lr.fit(X, y)
#dt = DecisionTreeClassifier(max_depth=5)
#dt.fit(X, y)

# Save the models and scaler
pickle.dump(lr, open('lr_model.pkl', 'wb'))
#pickle.dump(dt, open('dt_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

app = Flask(__name__)

# Define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)

    # Preprocess the data
    data = pd.DataFrame(data)
    data = data.drop(['customerID'], axis=1)
    data = scaler.transform(data)

    # Make the predictions
    lr_prediction = lr.predict(data)[0]
    #dt_prediction = dt.predict(data)[0]

    # Return the predictions as JSON
    output = {'lr_churn_prediction': int(lr_prediction)}#, 'dt_churn_prediction': int(dt_prediction)}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
