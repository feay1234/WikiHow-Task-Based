from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import threading

from urllib.parse import urljoin, urlparse
import sys
import pandas as pd
from gquestions import initBrowser, newSearch, crawlQuestions, prettyOutputName, flatten_csv
from time import sleep
import re


class MultiThread:

    def __init__(self, jobs, df):

        self.df = df
        self.pool = ThreadPoolExecutor(max_workers=10)
        self.finished_jobs = set([])
        self.to_do = jobs
        self.lock = threading.Lock()

    def write(self, job):

        id = job[0]
        start_time = job[1]
        end_time = job[2]

        tmp = self.df[self.df.AnonID == id]
        ses = '\t'.join(tmp[tmp['QueryTime'].between(start_time, end_time)]['Query'].tolist())
        #
        #
        self.lock.acquire()
        with open('data/test.csv', 'a+') as the_file:
            the_file.write('%s\n' % ses)
        self.lock.release()
        return

    def run_scraper(self):
        while True:
            try:
                job = self.to_do.get(timeout=10)
                # sleep(4)
                # print(self.to_do.qsize())
                if job not in self.finished_jobs:
                    # print(target_url)
                    self.finished_jobs.add(job)
                    self.pool.submit(self.write, job)
            except Empty:
                break

            except Exception as e:
                print(e)
                continue
if __name__ == '__main__':
    pass
    # s = MultiThread()
    # s.run_scraper()
    # run on 12:08


    #
    # import time
    # start = time.time()
    # crawl("how to clean house")
    # end = time.time()
    # print(end - start)
    #
    # start = time.time()
    # crawl("jarana manotumruksa")
    # end = time.time()
    # print(end - start)


#     start at 14:51


