from flask import Flask, render_template, request
from model import *
import pandas as pd


app = Flask(__name__)


@app.route("/")
def home():
    print(pd.__version__)
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    input_dict = request.form
    input_frame = pd.DataFrame(input_dict, index=[0])
    input_frame['fuel'] = input_frame['fuel'].apply(lambda x: fuel_value(x))
    input_frame['transmission'] = input_frame['transmission'].apply(lambda x: transmission_value(x))
    input_frame['seller_type'] = input_frame['seller_type'].apply(lambda x: seller_type(x))
    owner_arr = owner_onehot(input_frame['owner'].iloc[0])
    owner_arr_df = pd.DataFrame(owner_arr,
                                index=['First Owner', 'Fourth & Above Owner', 'Second Owner', 'Test Drive Car',
                                       'Third Owner']).T
    final_df = pd.concat([input_frame, owner_arr_df], axis=1)
    final_df = final_df.drop(['owner'], axis=1)
    prediction = model.predict(final_df)
    output = round(prediction[0], 2)
    result = 'Rs. ' + str(output)
    return render_template('index.html', prediction_text=result)


def owner_onehot(key):
    owner_dict = {
            'First Owner': [1, 0, 0, 0, 0],
            'Fourth & Above Owner': [0, 1, 0, 0, 0],
            'Second Owner': [0, 0, 1, 0, 0],
            'Test Drive Car': [0, 0, 0, 1, 0],
            'Third Owner': [0, 0, 0, 0, 1]
        }

    if key in owner_dict.keys():
        return owner_dict[key]


def fuel_value(key):
    fuel_dict = {
        'Petrol': 1,
        'Diesel': 0,
    }

    if key in fuel_dict.keys():
        return fuel_dict[key]
    else:
        return -1


def transmission_value(key):
    if key == 'Manual':
        return 1
    else:
        return 0


def seller_type(key):
    seller_dict = {
        'Individual': 1,
        'Dealer': 0
    }
    if key in seller_dict.keys():
        return seller_dict[key]
    else:
        return -1







if __name__ == "__main__":
    app.run(debug=True)


