import json
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from torch import embedding
import numpy as np

class Representation():
    """
    Representation is the base class for all representations.
    """
    
    def distance_metrics(self):
        return []

    def get_name(self):
        return ""

    def train(self):
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def load_data(self, path1 ,path2, targets):
        pass

    def get_embseddings(self, word):
        return []

    def compare(self, word, distance_metric = None):
        return 0

    def distance(self, word1, word2):
        return 0

    def save_results(self, dir_name, distance_metric):
        with open(f'{dir_name}/results_{distance_metric}.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Word", "Result", "Actual", "Expected"])
            for (word_it ,result_it,actual, exprected) in list(zip(self.words \
                    ,self.results\
                    ,self.actuals\
                    ,self.expecteds)) :
                writer.writerow([word_it, result_it, actual, exprected])

    def do_test(self, path):
        test_file = open(path)
        test_json = json.load(test_file)
        threshold = test_json["threshold"]

        self.load_data(test_json["corpora"][0], test_json["corpora"][1],[test["word"] for test in test_json["tests"]])
        
        if not "skip_training" in test_json:
            self.train()

        dict = {}
        dict['name'] = f"{self.get_name()}_{test_json['name']}"
        dict['language'] = test_json['language']
        dict['tesk_description'] = test_json['description']
        dict['distance_metrics'] = self.distance_metrics()
        dict['results'] = []
        dir_name = f'{test_json["target"]}/{self.get_name()}_{test_json["name"]}'

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for distance_metric in self.distance_metrics():
            self.actuals = []
            self.expecteds = []
            self.words = []
            self.results = []
            self.distances = []

            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            ranking_tests = False
            if "ranking_tests" in test_json:
                ranking_tests = True

            for test in test_json['tests']:
                word = test['word']
                
                result = self.compare(word, distance_metric)
                if result == None:
                    continue
                    
                self.distances.append(result)
            
            max_distance = np.max(self.distances)
            min_distance = np.min(self.distances)

            for distance in self.distances:
                if max_distance - min_distance != 0:
                    distance = (distance - min_distance) / (max_distance - min_distance)
        
                if distance >= threshold:
                    actual = 1.0
                else:
                    actual = 0.0
                
                if not ranking_tests:
                    expected = float(test['expected'])
                    if expected >= threshold:
                        expected_norm = 1.0
                    else:
                        expected_norm = 0.0
                    
                    if actual == expected_norm:
                        if actual == 1.0:
                            true_positives += 1
                        else:
                            true_negatives += 1
                    else:
                        if actual == 1.0:
                            false_positives += 1
                        else:
                            false_negatives += 1


                    self.expecteds.append(expected)

                self.results.append(distance)
                self.actuals.append(actual)
                self.words.append(word)

            if not ranking_tests:

                dict_distance_metric = {}
                dict_distance_metric['score'] = (true_negatives + true_positives) / (false_negatives + false_positives + true_negatives + true_positives)
                dict_distance_metric['metric'] = distance_metric
                dict_distance_metric['true_positives'] = true_positives
                dict_distance_metric['false_positives'] = false_positives
                dict_distance_metric['true_negatives'] = true_negatives
                dict_distance_metric['false_negatives'] = false_negatives
                
                dict['results'].append(dict_distance_metric)

                to_plot = {'Results': self.results,
                    'Expected': self.expecteds
                }

                plt.scatter(self.results, self.expecteds)
                plt.savefig(f'{dir_name}/{self.get_name()}_{test_json["name"]}_{distance_metric}_overall.png')
                plt.close()
            
            word_dict = {}
            for word,result in list(zip(self.words,self.results)):
                word_dict[word] = result
            sorted_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1])}
            plot_words = list(sorted_dict.keys())[-10:]
            plot_results = list(sorted_dict.values())[-10:]

            plt.bar(plot_words, plot_results, color ='blue')

            plt.xlabel("Word")
            plt.ylabel("Change")
            plt.xticks(rotation=30)
            plt.savefig(f'{dir_name}/{self.get_name()}_{test_json["name"]}_{distance_metric}_rank.png')
            plt.close()
            
            self.save_results(dir_name, distance_metric)

        with open(f'{dir_name}/words.csv', 'w') as file:
            writer = csv.writer(file)
            for test in test_json['tests']:
                word = test['word']
                embeddings = self.get_embseddings(word)
                for index, embedding in enumerate(embeddings):
                    row = [f'{word}_{index}']
                    for val in embedding:
                        row.append(val)
                    writer.writerow(row)
                
        desc_file = open(f"{dir_name}/desc.json", "w")
        json.dump(dict, desc_file, indent = 4)
        desc_file.close()