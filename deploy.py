from flask import jsonify
from flask import request
from flask import Flask
import pickle 


with open('D:\\Newfolder\\churn_model.pkl','rb') as f_in:
    model,dv=pickle.load(f_in)



# API
app=Flask('churn')
@app.route('/predict', methods=['POST'])
def predict():
    customer=request.get_json()
    X=dv.transform([customer])
    y_pred=model.predict_proba(X)[:,1]
    churn=y_pred>=0.5
    result={
        'churn probability': float(y_pred),
        'churn':bool(churn)
    }

    return jsonify(result)

if __name__ =="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


# open Postman then write any user  info in the body(sending request) like this 
""" {
    "customerid": "8879-zkjof",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85,
    "totalcharges": 3320.75
} """