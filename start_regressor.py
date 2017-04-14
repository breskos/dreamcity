import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
import sys
import pickle
import loc_features as features

REGRESSOR_FILE = 'regressor.p'
MODEL_FILE = 'model.p'

regressor = pickle.load(open(REGRESSOR_FILE, "rb" ))
model = pickle.load(open(MODEL_FILE, "rb"))

location = sys.argv[1]
vector = features.get_features(model, location)
predict = regressor.predict([vector])

print (location)
print ("lat: " + str(predict[0][1]) + " lng: " + str(predict[0][0]))   
