import json
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from torch import embedding

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
        
        if "model_path" in test_json:
            self.load_model(test_json["model_path"])
        else:
            self.train()

        dict = {}
        dict['name'] = self.get_name()
        dict['language'] = test_json['language']
        dict['tesk_description'] = test_json['description']
        dict['distance_metrics'] = self.distance_metrics()
        dir_name = f'{test_json["target"]}/{self.get_name()}_{test_json["name"]}'

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for distance_metric in self.distance_metrics():
            max_score = 0
            score = 0

            self.actuals = []
            self.expecteds = []
            self.words = []
            self.results = []

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
                
                if result >= threshold:
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
                        score += expected
                    max_score += expected
                    
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

                self.results.append(result)
                self.actuals.append(actual)
                self.words.append(word)

            if not ranking_tests:

                dict[distance_metric] = {}
                dict[distance_metric]['score'] = score
                dict[distance_metric]['max_score'] = max_score
                dict[distance_metric]['true_positives'] = true_positives
                dict[distance_metric]['false_positives'] = false_positives
                dict[distance_metric]['true_negatives'] = true_negatives
                dict[distance_metric]['false_negatives'] = false_negatives
                
                if not "ignore_spearmanr" in test_json:
                    sp = stats.spearmanr(self.results, self.expecteds)
                    dict[distance_metric]['spearmanr_pvalue'] = sp.pvalue
                    dict[distance_metric]['spearmanr_correlation'] = sp.correlation

                to_plot = {'Results': self.results,
                    'Expected': self.expecteds
                }

                df = pd.DataFrame(to_plot,columns=['Results','Expected'])
                normalized_df=(df-df.mean())/df.std()
                normalized_df.plot(x ='Results', y='Expected', kind = 'scatter')
                plt.savefig(f'{dir_name}/{self.get_name()}_{distance_metric}_overall.png')
            
            word_dict = {}
            for word,result in list(zip(self.words,self.results)):
                word_dict[word] = result
            sorted_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1])}
            plot_words = list(sorted_dict.keys())[-10:]
            plot_results = list(sorted_dict.values())[-10:]

            if not ranking_tests:
                plt.bar(plot_words, plot_results, color ='blue')
            else:
                for word in plot_words:
                    i = self.words.index(word)
                    if self.actuals[i] == 1:
                        plt.bar(word, result, color ='green')
                    else:
                        plt.bar(word, result, color ='red')

            plt.xlabel("Word")
            plt.ylabel("Change")
            plt.xticks(rotation=30)
            plt.savefig(f'{dir_name}/{self.get_name()}_{distance_metric}_rank.png')
            
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