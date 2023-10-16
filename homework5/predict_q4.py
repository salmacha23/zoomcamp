import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as model_f_in:
    model = pickle.load(model_f_in)
    
with open(dv_file, 'rb') as dv_f_in:
    dv = pickle.load(dv_f_in)

app = Flask("credit scoring")

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    credit_decision = y_pred >= 0.5

    result = {
        'credit_probability': float(y_pred),
        'credit_decision': bool(credit_decision)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)