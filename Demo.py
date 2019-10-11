import pandas as pd
import os

from run_scraper import crawl

if __name__ == '__main__':

    crawl("how to cook pasta", "csv/", "", False)
