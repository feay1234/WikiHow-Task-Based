from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import sys
import json

import time
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from selenium.webdriver.chrome import webdriver
import os

from tqdm import tqdm

from gquestions import initBrowser, newSearch, crawlQuestions, prettyOutputName, flatten_csv
from time import sleep
import re

def crawl(keyword, folder, cat):
    # args = docopt(usage)
    args = {'--csv': True,
            '--headless': True,
            '--help': False,
            '<depth>': None,
            '<keyword>': keyword,
            'depth': False,
            'en': True,
            'es': False,
            'query': True}
    # print(args)
    MAX_DEPTH = 1

    if args['<depth>']:
        depth = int(args['<depth>'])
        if depth > MAX_DEPTH:
            sys.exit("depth not allowed")
    else:
        depth = 0

    if args['en']:
        lang = "en"
    elif args['es']:
        lang = "es"

    if args['<keyword>']:
        if args['--headless']:
            browser = initBrowser(True)
        else:
            browser = initBrowser()
        query = args['<keyword>']
        start_paa = newSearch(browser, query, lang)

        initialSet = {}
        cnt = 0
        for q in start_paa:
            initialSet.update({cnt: q})
            cnt += 1

        paa_list = []

        crawlQuestions(query, lang, browser, start_paa, paa_list, initialSet, depth)

        if paa_list[0]['children']:
            _path = folder + prettyOutputName(query, 'csv')
            flatten_csv(paa_list, depth, _path)
        else:
            with open(folder+cat, 'a') as the_file:
                the_file.write(query+'\n')

        browser.close()

        return start_paa

        # get more queries
        # initialSet = {}
        # cnt = 0
        # for q in start_paa:
        #     initialSet.update({cnt: q})
        #     cnt += 1
        # paa_list = []
        # crawlQuestions(start_paa, paa_list, initialSet, query, browser, depth)




        # do not need this one
        # treeData = 'var treeData = ' + json.dumps(paa_list) + ';'
        # if paa_list[0]['children']:
        #     root = os.path.dirname(os.path.abspath(__file__))
        #     templates_dir = os.path.join(root, 'templates')
        #     env = Environment(loader=FileSystemLoader(templates_dir))
        #     template = env.get_template('index.html')
        #     filename = os.path.join(root, 'html', prettyOutputName())
        #     with open(filename, 'w') as fh:
        #         fh.write(template.render(
        #             treeData=treeData,
        #         ))

    # if args['--csv']:
    #     if paa_list[0]['children']:
    #         _path = 'tmp/' + prettyOutputName(query, 'csv')
    #         flatten_csv(paa_list, depth, _path)

class MultiThreadScraper:

    def __init__(self, to_crawl, folder, cat):

        # self.base_url = base_url
        # self.root_url = '{}://{}'.format(urlparse(self.base_url).scheme, urlparse(self.base_url).netloc)
        self.pool = ThreadPoolExecutor(max_workers=20)
        self.scraped_pages = set([])
        self.to_crawl = to_crawl
        self.folder = folder
        self.cat = cat

        # regex = re.compile('[^a-zA-Z]')
        # df = pd.read_csv("data/articles.txt", error_bad_lines=False).values.tolist()
        # unique_set = set([])
        # for i in df:
        #     _ = regex.sub(' ', i[0])
        #     unique_set.add(_)
        # for i in unique_set:
        #     self.to_crawl.put(i)
        # # self.to_crawl.put("how to cook pasta")



    # def parse_links(self, html):
    #     self.to_crawl.put("www.google.com")
    #
    # def scrape_info(self, html):
    #     return
    #
    # def post_scrape_callback(self, res):
    #     result = res.result()
    #     if result and result.status_code == 200:
    #         self.parse_links(result.text)
    #         self.scrape_info(result.text)

    def scrape_page(self, url):
        try:
            crawl(url, self.folder, self.cat)
        except Exception as e:

            with open(folder+cat, 'a') as the_file:
                the_file.write(url+'\n')

        return

    def run_scraper(self):

        for i in tqdm(range(len(self.to_crawl))):
            try:
                target_url = self.to_crawl[i]
                self.pool.submit(self.scrape_page, target_url)
                sleep(10)

            except Exception as e:
                print(e)
                continue

if __name__ == '__main__':

    # crawl("how to cook pasta", "csv/")

    all_categories = "Arts and Entertainment·Cars & Other Vehicles·Computers and Electronics·Education and Communications·Family Life·Finance and Business·Food and Entertaining·Health·Hobbies and Crafts·Holidays and Traditions·Home and Garden·Personal Care and Style·Pets and Animals·Philosophy and Religion·Relationships·Sports and Fitness·Travel·Work World·Youth"
    all_categories = all_categories.lower()
    all_categories = all_categories.split("·")
    done = ['sports and fitness', 'health', 'education and communications', 'arts and entertainment']
    for i in done:
        all_categories.remove(i)

    df = pd.read_csv("data/wikihowSep.csv")
    df = df[df.sectionLabel != "Steps"]
    df["title"] = df["title"].str.lower()
    df["sectionLabel"] = df["sectionLabel"].str.lower()

    wikiCat = pd.read_csv("data/cate.csv", sep=",", error_bad_lines=False, names=["title", "category"])
    wikiCat['title'] = wikiCat['title'].str.replace("https://www.wikihow.com/", "").str.replace("%22", "").str.replace(
        "-", " ").str.lower()
    wikiCat['title'] = ["how to " + i for i in wikiCat['title'].tolist()]
    wikiCat['category'] = wikiCat['category'].str.lower()

    for cat in all_categories[::-1]:
        print(cat)
        tasks = wikiCat[wikiCat.category.str.contains(cat)].title.tolist()

        cat = cat.replace(" ", "_")

        folder = "csv/%s/" % (cat)
        if not os.path.exists(folder):
            os.makedirs(folder)

        finish = []
        for file in os.listdir(folder):
            if os.stat(folder + file).st_size > 0:
                if ".csv" in file:
                    finish.append(file.split(".csv")[0].replace("_", " "))

        if os.path.exists(folder+cat):
            finish.extend([line.rstrip('\n') for line in open(folder+cat)])



        # df["headline"] = df["headline"].str.lower().str.replace("\n", "").str.replace(".","")
        # df["title"] = df["title"].str.lower()
        # print(tasks)

        queries = []
        for name, row in df[df.title.isin(tasks)].drop_duplicates(["sectionLabel"]).groupby("title"):
            queries.append(name)
            queries.extend(row.sectionLabel.tolist())

        to_crawl = []
        for t in queries:
            if t not in finish:
                to_crawl.append(t)


        print("Finished: %d To crawl: %d" % (len(finish), len(to_crawl)))
        if len(to_crawl) > 0:
            s = MultiThreadScraper(to_crawl, folder, cat)
            s.run_scraper()
        # break
