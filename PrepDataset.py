
import pandas as pd
import numpy as np
import random

df = pd.read_csv("data/d3_wikihow.csv")
task_nb = df.Task.unique()
for idx, row in df[df.Source == "google"].iterrows():
    pos_task = row.Task

    # random negative task
    neg_task = random.choice(task_nb)
    while neg_task == pos_task:
        neg_task = random.choice(task_nb)

    pos_article = df[(df.Task == pos_task) & (df.Source == "wikihow")].sample(1).Query.values[0]
    neg_article = df[(df.Task == neg_task) & (df.Source == "wikihow")].sample(1).Query.values[0]
    #     print(pos_article.Query.values[0])
    #     print(neg_article.Query.values[0])

    with open('data/query_task_prediction2.csv', 'a') as the_file:
        the_file.write('%s' % row.Query.strip() + "\t" + pos_article + "\t1\n")
        the_file.write('%s' % row.Query.strip() + "\t" + neg_article + "\t0\n")