import requests
import main
import joblib
import pandas as pd


#with open("model_fitted.pkl", 'rb') as file:
#    model = joblib.load(file)


#data = main.ModelRequestData(total_square = 100.0, rooms = 1, floor = 1, distance_to_center = 0.0)
#input_data = data.dict()
    #input_data = {'total_square': total_square, 'rooms':rooms, 'floor':floor, 'distance_to_center': distance_to_center}
#input_df = pd.DataFrame(input_data, index=[0])
#result = model.predict(input_df)[0]
a = requests.get('http://127.0.0.1:8000/predict_get', params = {'total_square': 200.0, 'rooms':'azaza', 'floor':1, 'distance_to_center': 0.0})
print(a.json())

#print(result)
