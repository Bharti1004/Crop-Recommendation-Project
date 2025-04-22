from flask import Flask,request,render_template

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import pickle

# importing our model
model=pickle.load(open('model.pkl','rb'))

# creating Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = int(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # scaled_features = ms.transform(single_pred)
    # final_features = sc.transform(scaled_features)
    prediction = model.predict(single_pred)
    
    crop_dict={1:'rice',2:'maize',3:'jute',4:'cotton',5:'coconut',6:'papaya',7:'orange',8:'apple',9:'muskmelon',  #yha pr hmne no. ko as a key
           10:'watermelon',11:'grapes',12:'mango',13:'banana',14:'pomegranate',15 :'lentil',
           16:'blackgram',17:'mungbean',18:'mothbeans',19:'pigeonpeas',20:'kidneybeans',21:'chickpea',22:'coffee'}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        
    return render_template('index.html',result = result)

# python-main
if __name__== "__main__":
    app.run(debug=True)