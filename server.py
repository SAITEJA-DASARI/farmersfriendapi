from flask import Flask, request, url_for, flash, jsonify
from flask_cors import CORS
import numpy as np
import pickle as p
import json

app=Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
modelfile='models/final_prediction.pickle'
model=p.load(open(modelfile,'rb'))
@app.route("/api/predict",methods=['post'])
def makecalc():
    data=request.get_json()
    temp=data["temperature"]
    hum=data["humidity"]
    ph=data["ph"]
    rain=data["rainfall"]
    l=[]
    l.append(temp)
    l.append(hum)
    l.append(ph)
    l.append(rain)
    predictCrop=[l]
    # print(predictCrop)
    #scaling the data
    from myModel import standardisation
    predictCrop=standardisation.transform(predictCrop)
    # from sklearn import preprocessing
    # min_max_scaler=preprocessing.MinMaxScaler(feature_range =(0,1))
    # predictCrop=min_max_scaler.fit_transform(predictCrop)
    # print(predictCrop)
    predictions=model.predict(predictCrop)
    predicted_crop=predictions[0]
    return jsonify(predicted_crop)

if __name__ == '__main__':
    
    app.run(debug=True,host='0.0.0.0',port=5500)
    
