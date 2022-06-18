from sklearn.manifold import TSNE

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv

keys = []
word_clusters = []

with open('demo/data/sgns_op_eng_sem_change_description/words.csv', 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    sup = 0
    for row in csv_reader:
        keys.append(row[0])
        word_clusters.append([row[1:]])
        sup += 1
        if sup == 40:
            break

def tsne_plot_similar_words( embedding_clusters):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    for embeddings, word, in zip( embedding_clusters, keys):
        x = embeddings[:,0][0]
        y = embeddings[:,1][0]
        z = embeddings[:,2][0]
        ax.scatter3D(x, y, z, c='b')
        ax.text(x, y, z, word)
    plt.grid(True)
    plt.show()

tsne_model_en_2d = TSNE(n_components=3, learning_rate='auto', init='random')
word_clusters = np.array(word_clusters)
n, m, k = word_clusters.shape
embeddings_en_3d = np.array(tsne_model_en_2d.fit_transform(word_clusters.reshape(n * m, k))).reshape(n, m, 3)


tsne_plot_similar_words(embeddings_en_3d)