from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
# load the model
model2 = pickle.load(open("model2.pkl","rb"))

@app.route("/")
def home():
    return render_template('index2.html')
@app.route("/predict",methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction1 = model2.predict(features)
    prediction = prediction1[[0]]
    return render_template("/index2.html",prediction_text = 'The average house price is ${}k'.format(prediction))

if __name__ =='__main__':
    app.run(debug=True)