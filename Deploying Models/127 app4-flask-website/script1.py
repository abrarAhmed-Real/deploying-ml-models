from flask import Flask, render_template , request
import pickle
model=pickle.load(open('model.pkl' , 'rb'))
Spam_model=pickle.load(open('spam_model.pkl' , 'rb'))
import numpy as np
app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/linear regression')
def lr():
        return render_template("lr.html")
@app.route('/predict' , methods=["POST" , "GET"])
def predict():
    if request.method=="POST":
        values=[int(x) for x in request.form.values()]
        features=[np.array(values)]
        pred=int(model.predict(features))
        return render_template("lr.html" , pred_text= str(pred) + " $" )        

@app.route('/logistic regression')
def lg():
    return render_template("lg.html")


@app.route('/svm')
def svm():
    return render_template("svm.html")

@app.route('/spam_detector')
def spam():
        return render_template("spam.html")

@app.route('/spam_detect' , methods=["GET" , "POST"])
def spam_detect():
    if request.method=="POST":
        msg=request.form["msg"]
        pred=Spam_model.predict([msg])
        return render_template("spam.html" , pred_text= str(pred[0]))
if __name__=="__main__":
    app.run(debug=True)
