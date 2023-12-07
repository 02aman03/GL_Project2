from flask import Flask, request, Response
from joblib import load, dump
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
warnings.filterwarnings("ignore")
my_model = load('my_model.joblib')
le = load('label_encoder.joblib')
#preds = my_model.predict([[1.8,2.4,3.5,4.3], [3.6,2.7,1.4,2.5], [5.2,6.1,3.8,2.9], [1.3,3.2,3.1,4.0]])

def create_df(x):
    if np.array(x).ndim == 1:
        if np.shape(x)[0] == 4:
            temp_dict = {'sepal.length':[x[0]], 'sepal.width':[x[1]], 'petal.length':[x[2]], 'petal.width':[x[3]]}
            temp_df = pd.DataFrame(temp_dict)
            return temp_df
        else:
            print("Invalid Input, check the number of entries you passed in the list.")
    elif np.array(x).ndim == 2:
        try:
            nrows = np.shape(x)[0]
            ncols = np.shape(x)[1]
            if np.shape(x)[0] == 4 and np.shape(x)[1] != 4:
                temp_dict = {'sepal.length':x[0], 'sepal.width':x[1], 'petal.length':x[2], 'petal.width':x[3]}
                temp_df = pd.DataFrame(temp_dict)
            elif np.shape(x)[1] == 4 and np.shape(x)[0] != 4:
                x = np.transpose(x)
                temp_dict = {'sepal.length':x[0], 'sepal.width':x[1], 'petal.length':x[2], 'petal.width':x[3]}
                temp_df = pd.DataFrame(temp_dict)
            elif np.shape(x)[1] == 4 and np.shape(x)[0] == 4:
                temp_dict = {'sepal.length':x[0], 'sepal.width':x[1], 'petal.length':x[2], 'petal.width':x[3]}
                temp_df = pd.DataFrame(temp_dict)
                print("Input ambiguous, since both the number of rows and columns is equal, function will assume it as a standard input.")
            return temp_df
        except:
            print("The input data is invalid. Please check if all the rows have the appropriate amount of entries.")
    else:
        print("Invalid Input, it cannot be more than 2-Dimensional.")

def create_df(x):
    if np.array(x).ndim == 1:
        if np.shape(x)[0] == 4:
            temp_dict = {'sepal.length':[x[0]], 'sepal.width':[x[1]], 'petal.length':[x[2]], 'petal.width':[x[3]]}
            temp_df = pd.DataFrame(temp_dict)
            return temp_df
        else:
            print("Invalid Input, check the number of entries you passed in the list.")
    elif np.array(x).ndim == 2:
        try:
            nrows = np.shape(x)[0]
            ncols = np.shape(x)[1]
            if np.shape(x)[0] == 4 and np.shape(x)[1] != 4:
                temp_dict = {'sepal.length':x[0], 'sepal.width':x[1], 'petal.length':x[2], 'petal.width':x[3]}
                temp_df = pd.DataFrame(temp_dict)
            elif np.shape(x)[1] == 4 and np.shape(x)[0] != 4:
                x = np.transpose(x)
                temp_dict = {'sepal.length':x[0], 'sepal.width':x[1], 'petal.length':x[2], 'petal.width':x[3]}
                temp_df = pd.DataFrame(temp_dict)
            elif np.shape(x)[1] == 4 and np.shape(x)[0] == 4:
                temp_dict = {'sepal.length':x[0], 'sepal.width':x[1], 'petal.length':x[2], 'petal.width':x[3]}
                temp_df = pd.DataFrame(temp_dict)
                print("Input ambiguous, since both the number of rows and columns is equal, function will assume it as a standard input.")
            return temp_df
        except:
            print("The input data is invalid. Please check if all the rows have the appropriate amount of entries.")
    else:
        print("Invalid Input, it cannot be more than 2-Dimensional.")


def user_function(data_points):
    try:
        x_data = create_df(data_points)
        preds = my_model.predict(x_data)
        if preds.shape[0] == 1:
            return le.inverse_transform(preds)[0]
        return le.inverse_transform(preds)
    except:
        print('The input needs to be in a list of list format of the type [[x11,x21,x31,...], [x12,x22,x32,...], [x13,x23,x33,...], [x14,x24,x34,...]]')


app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def hello_world():
    return "Hello World"

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    data = request.json
    user_sent_data = data.get('my_data')
    print(type(user_sent_data))
    user_sent_list = eval(user_sent_data)
    return Response(str(user_function(user_sent_list)))

if __name__ == '__main__':
    app.run(debug=True)