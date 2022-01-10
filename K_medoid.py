import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import random
from multiprocessing import Process
import tkinter as tk
from tkinter import filedialog
import MovieApp

def K_medoid(data, k, prev_cost, count, clusters=None, medoids=None):
    cluster_sum = 0
    random.seed(0)
    while True:
        if medoids is None or not medoids:
            medoids = random.sample(data, k)

        else:
            random.shuffle(medoids)

            for _ in range(0, int(k/2)):
                medoids.pop()
            medoids += random.sample(data, int(k/2))
        clusters = defaultdict(list)

        for item in data:
            temp = []
            for i in range(0, len(medoids)):
                med = medoids[i]
                if med is None or not med:
                    break
                else:
                    temp.append(np.linalg.norm(med[0]-item[0])+np.linalg.norm(med[1]-item[1]))
            min_index = np.argmin(temp)
            clusters[min_index].append(item)

        for i in range(0, len(medoids)):
            inter_cluster = clusters[i]
            for j in range(0, len(inter_cluster)):
                item_cluster = inter_cluster[j]
                medoid = medoids[i]
                cluster_sum += (np.linalg.norm(medoid[0]-item_cluster[0]) +
                                np.linalg.norm(medoid[1]-item_cluster[1]))

        if cluster_sum < prev_cost:
            prev_cost = cluster_sum
        else:
            break

        count += 1
    return clusters


def plot_graph(clusters):
    markers = ['bo', 'go', 'ro', 'c+', 'm+', 'y+']
    for i in range(0, len(clusters.keys())):
        data = clusters.get(i)
        for j in range(0, len(data)):
            df = data[j]
            plt.plot(df[0], df[1], markers[i])

    plt.xlabel('IMDb Scores')
    plt.ylabel('Gross')
    plt.title('K-Medoid clusters')
    plt.legend()
    plt.show()


def assign_target(row, clusters):

    x = row['movie_title']

    for i in range(0, len(clusters.keys())):
        data = clusters.get(i)
        for j in range(0, len(data)):
            df = data[j]
            if df[2] == x:
                row['cluster'] = 'cluster' + str(i)

    return row


def open_cv(root):
    root.filename = filedialog.askopenfilename(parent=root, filetypes=[("All files","*.csv")])
    print(root.filename)


def gui(df):
    #start
    root = tk.Tk()
    editor = MovieApp(root, df)
    root.mainloop()


def init_app():

    columns = ['movie_title', 'num_user_for_reviews', 'budget', 'num_critic_for_reviews',
               'movie_facebook_likes', 'num_voted_users', 'duration', 'gross', 'imdb_score']

    #loading dataset
    df = pd.read_csv('movie_metadata.csv').dropna(axis=0).reset_index(drop=True)

    p = Process(target=gui, args=(df[columns],))
    p.start()

    #choosing features and running k-medoids
    dataset = df[['gross', 'imdb_score', 'movie_title']]
    dataset = dataset.values.tolist()
    clusters = K_medoid(dataset, 5, np.inf, 0)
    print(len(clusters))

    #plot cluster graph
    p = Process(target=plot_graph, args=(clusters,))
    p.start()


if __name__=='__main__':
    init_app()

