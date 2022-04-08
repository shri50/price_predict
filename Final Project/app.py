
from flask import Flask,request,render_template
import json
import pickle
import numpy as np

sqft_max=52272.0
sqft_min=1.0
bath_max=14.0
bath_min=1.0
bhk_max=11
bhk_min=1


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods =['POST','GET'])
def predict():

    total_sqft = int(request.form.get('total_sqft'))
    bath = int(request.form.get('bath'))
    bhk = int(request.form.get('bhk'))
    area_type = request.form.get('area_type')
    location = request.form.get('location')
    
    with open(r'artifacts\columns.json','r') as file:
        column_dict = json.load(file)

    column_list = column_dict['data_columns']

    user_data = np.zeros(len(column_list))
    
    
    
    
    scaled_sqft = total_sqft - sqft_min/ sqft_max-sqft_min
    scaled_bath = bath - bath_min/ bath_max-bath_min
    sacaled_bhk = bhk - bhk_min/ bhk_max-bhk_min
    
    user_data[0] = scaled_sqft
    user_data[1] = scaled_bath
    user_data[2] = sacaled_bhk
    
    
    area_type_index = column_list.index(area_type)
    user_data[area_type_index] = 1
    
    location_index = column_list.index(location)
    user_data[location_index] = 1
    

    with open(r'artifacts\model.pickle','rb') as model_file:
        model = pickle.load(model_file)
    
    result = model.predict([user_data])[0]
    print(result)




    return render_template('index.html',predict_value=result)



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)