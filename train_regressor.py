import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
import loc_features as features
import sys
import pickle

DATA_FILE = 'data/locations.csv'
NGRAMS = 3
HIDDEN_LAYER_SIZE = 100
SOLVER = "adam"
LEARNING_RATE = "adaptive"
ACTIVIATION = "logistic"
MAXIMUM_ITERATIONS = 900000

with open(DATA_FILE) as f:
    content = f.read().splitlines()

# preprocessing of training data
X = []
names = []
for row in content:
    data = row.split('","')
    v = [float(data[3]), float(data[2])]
    if v[0] != 0.0 and v[1] != 0:
        X.append(np.array(v))
        names.append(data[1])
X = np.array(X)

# build n-gram feature model here
model = features.build_features(NGRAMS, names)
# get features from created model
feature_vectors = []
for i in range(0, len(names)):
    feature_vectors.append(features.get_features(model, names[i]))

# create neural net regressor
reg = MLPRegressor(
    hidden_layer_sizes=(HIDDEN_LAYER_SIZE,),
    activation=ACTIVIATION,
    learning_rate=LEARNING_RATE,
    solver=SOLVER,
    early_stopping=False,
    max_iter=MAXIMUM_ITERATIONS
    )
# train neural net regressor
reg.fit(feature_vectors,X)

# test prediction
test_locations = ['Berlin', 'Hamburg', 'Stuttgart','Cottbus', 'Kiel', 'Dortmund', 'Bremen']
test_x=[
    features.get_features(model, test_locations[0]),
    features.get_features(model, test_locations[1]),
    features.get_features(model, test_locations[2]),
    features.get_features(model, test_locations[3]),
    features.get_features(model, test_locations[4]),
    features.get_features(model, test_locations[5]),
    features.get_features(model, test_locations[6])
]

predict=reg.predict(test_x)
print "_Input_\t_output_"
for i in range(len(test_x)):
    print "  ", test_locations[i] , "---->" , predict[i]

pickle.dump(model, open("model.p", "wb"))
pickle.dump(reg, open("regressor.p", "wb"))
