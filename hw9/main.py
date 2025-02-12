import os

import pandas as pd
import streamlit as st
import folium as fl
from streamlit_folium import st_folium

from src.utils import prepare_data, train_model, read_model, distance_to_center

lat, lon = 55.75222, 37.61556

st.set_page_config(
    page_title="Appartment price estimator",
)

model_path = 'model_fitted.pkl'

total_square = st.sidebar.number_input("Total square of an appartment", 0, 3000, 30)

floor = st.sidebar.number_input(
    "Floor",
    1, 100, 1,
)
rooms = st.sidebar.number_input(
    "Rooms",
    1,20,1,
)



if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)

model = read_model('model_fitted.pkl')







#def get_pos(lat, lng):
#    return lat, lng
st.header('Appartment Price Estimator')
'Click on the location of the appartment'
m = fl.Map(location = (55.75222, 37.61556))

# Initialize a variable to store the current marker
current_marker = None

# Function to add a marker
#def add_marker(lat, lon):
#    global current_marker
#    if current_marker:
#        # Remove the previous marker
#        m.remove_child(current_marker)

    # Create a new marker
#    current_marker = fl.Marker(location=[lat, lon])
#    m.add_child(current_marker)

# Click event handler
m.add_child(
    fl.LatLngPopup()
)

# Streamlit app logic
map_data = st_folium(m, height=350, width=700)

if map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    # Add marker at the clicked location
    #add_marker(lat, lon)
    # Display the last clicked position
    #st.write(f'Last clicked position is {lat}, {lon}')
# create input DataFrame
inputDF = pd.DataFrame(
    {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        'distance_to_center': distance_to_center(lat, lon),
    },
    index=[0],
)

if st.sidebar.button('Estimate price!'):
    preds = model.predict(inputDF)
    preds = round(preds[0]*10e-7, 1)
    t = f'The price based on the information provided is: **{preds} million roubles**'
    st.markdown(t)
    if preds <= 0: st.markdown('stooooooopid model :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:')
