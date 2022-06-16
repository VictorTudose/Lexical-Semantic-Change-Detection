from representations.representation import Representation

import csv
import pandas as pd
from sklearn.cluster import AffinityPropagation


class ELMo(Representation):

    def __init__(self):
        self.words = [[], []]
        self.clusterings = [{}, {}]

    def get_name(self):
        return "elmo"
    
    def load_corpus(self, path, index):
        words = {}
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = list(row.values())
                if row[0] not in words:
                    words[row[0]] = []
                words[row[0]].append(row[1:])

        for word in words:
            words[word] = pd.DataFrame(words[word])
        self.words[index] = words

        for word in self.words[index]:
            X = []
            for val in self.words[index][word].values:
                X.append(val)    
            clustering = AffinityPropagation(random_state=9).fit(X)
            self.clusterings[index][word] = clustering.labels_
    
    def load_data(self, path1 ,path2, _):
        self.load_corpus(path1, 0)
        self.load_corpus(path2, 1)
    
    def compare(self, word, distance_metric = None):
        if distance_metric is None:
            distance_metric = "euclidean"
        if word not in self.words[0]:
            return 0
        if word not in self.words[1]:
            return 0
        