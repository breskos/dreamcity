from sklearn.feature_extraction.text import CountVectorizer


def preprocess(name):
    strip = ['(',')','-']
    name = '$' + (name.replace(' ', '$')).lower()  + '$'
    for s in strip:
        name = name.replace(s, '')
    return name

def build_features(types, names):
    vector = []
    for name in names:
        name = preprocess(name)
        if types == 3:
            grams = find_trigrams(name)
        else:
            grams = find_bigrams(name)
        for g in grams:
            if g not in vector:
                vector.append(g)
    return vector

def get_features(features, name):
    grams = find_trigrams(preprocess(name))
    vector = [0] * len(features)
    for gram in grams:
        try:
            idx = features.index(gram)
        except:
            print('not found')
        else:
            vector[idx] = 1
    return vector

def find_trigrams(name):
  trigrams = []
  for i in range(len(name)-2):
      trigrams.append(name[i] + name[i+1] + name[i+2])
  return trigrams

def find_bigrams(name):
    bigrams = []
    for i in range(len(name) - 1):
        bigrams.append(name[i] + name[i + 1])
    return bigrams