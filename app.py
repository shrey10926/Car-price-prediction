from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        Power = request.form['powerPS']
        Model = request.form['model']
        Kilometer = request.form['kilometer']
        FuelType = request.form['fuelType']
        VehicleType = request.form['vehicleType']
        Gearbox = request.form['gearbox']
        NotRepairedDamage = request.form['notRepairedDamage']
        Brand = request.form['brand']
        Age = request.form['age']
        
        data = [[Power, Model, Kilometer,FuelType, VehicleType, Gearbox, 
                 NotRepairedDamage, Brand, Age]]
        
        input_variables = pd.DataFrame(data,columns=['powerPS', 'model', 'kilometer', 'fuelType', 
                                                     'vehicleType','gearbox', 'notRepairedDamage', 
                                                     'brand', 'age'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return render_template('index.html', original_input={'powerPS': Power, 'model': Model, 
                                                             'kilometer': Kilometer, 'fuelType': FuelType, 
                                                             'vehicleType': VehicleType, 'gearbox': Gearbox, 
                                                             'notRepairedDamage': NotRepairedDamage, 'brand': Brand, 'age':Age},
                                     result=predictions)

if __name__ == '__main__':
    app.run(debug = True)
