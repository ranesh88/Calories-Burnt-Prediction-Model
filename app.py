from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler
import sklearn
import pickle
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('rf.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
calories=pd.read_csv('calories_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    Gender=sorted(calories['Gender'].unique())

    Gender.insert(0,'Select Gender')

    return render_template('index.html',Gender=Gender)

@app.route('/predict',methods=['POST'])
@cross_origin()

def predict():
    Gender  = request.form.get('Gender')
    Age = request.form.get('Age')
    Height = request.form.get('Height')
    Weight = request.form.get('Weight')
    Duration = request.form.get('Duration')
    Heart_Rate = request.form.get('Heart_Rate')
    Body_Temp = request.form.get('Body_Temp')

    features = np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]],dtype=object)
    transformed_features = preprocessor.transform(features)
    prediction = model.predict(transformed_features).reshape(1, -1)

    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__=='__main__':
    app.debug = True
    app.run()

