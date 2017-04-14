import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
import sys
import pickle
import loc_features as features
from math import radians, cos, sin, asin, sqrt

DATA_FILE = 'data/locations.csv'
REGRESSOR_FILE = 'regressor.p'
MODEL_FILE = 'model.p'

regressor = pickle.load(open(REGRESSOR_FILE, "rb" ))
model = pickle.load(open(MODEL_FILE, "rb"))

with open(DATA_FILE) as f:
    content = f.read().splitlines()
# preprocessing
X = []
names = []
vector = []
for row in content:
    data = row.split('","')
    v = [float(data[3]), float(data[2])]
    if v[0] != 0.0 and v[1] != 0:
        X.append(np.array(v))
        names.append(data[1])
        vector.append(features.get_features(model, data[1]))
X = np.array(X)

predictions = regressor.predict(vector)
n = 0
s = 0
for i in range(len(vector)):
    n += 1
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [predictions[i][0], predictions[i][1], X[i][0], X[i][1]])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    s += c * r

# evaluation results
print ("sum of all drifts = " + str(s))
print ("number of samples = " + str(n))
print ("mean drift in kilometers = " + str(s / n) + " km")
