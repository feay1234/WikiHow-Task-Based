import pandas as pd
import numpy as np
import random
from string import digits
import re


def query_task_prediction():
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

        with open('data/query_task_prediction2.csv', 'a') as the_file:
            the_file.write('%s' % row.Query.strip() + "\t" + pos_article + "\t1\n")
            the_file.write('%s' % row.Query.strip() + "\t" + neg_article + "\t0\n")

def step_prediction():
    wiki = pd.read_csv("data/wikihowSep.csv")
    wiki = wiki.dropna()
    wiki['headline'] = [i.strip() for i in wiki['headline']]

    remove_digits = str.maketrans('', '', digits)
    wiki['title'] = [i.translate(remove_digits) for i in wiki['title']]
    for col in wiki:
        wiki[col] = [re.sub(r'[\W_]+', ' ', i).lower() for i in wiki[col]]

    singleWiki = wiki[wiki['sectionLabel'] == "steps"]
    for name, group in singleWiki.groupby("title"):
        for i in range(len(group) - 1):
            with open('data/step_prediction.csv', 'a') as the_file:
                the_file.write(
                    group.iloc[i].headline + "\t" + group.iloc[i + 1].headline + "\t" + group.iloc[i].text + "\t" +
                    group.iloc[i + 1].text + "\t" + name + "\n")

    singleWiki = wiki[wiki['sectionLabel'] == "steps"]
    tasks = singleWiki.title.unique()
    for name, group in singleWiki.groupby("title"):
        for idx, row in group.iterrows():
            pos_task = row.sectionLabel if row.sectionLabel != "steps" else row.title
            step = row.headline
            text = row.text

            neg_task = random.choice(tasks)
            while neg_task == pos_task:
                neg_task = random.choice(tasks)

            with open('data/step_task_prediction.csv', 'a') as the_file:
                the_file.write(step +"\t"+ text + "\t" + pos_task + "\t1\n")
                the_file.write(step +"\t"+ text + "\t" + neg_task + "\t0\n")
