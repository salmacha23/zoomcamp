import pickle 

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as model_f_in:
    model = pickle.load(model_f_in)
    
with open(dv_file, 'rb') as dv_f_in:
    dv = pickle.load(dv_f_in)
    
customer = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)