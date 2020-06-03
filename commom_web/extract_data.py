import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from langdetect import detect
import gzip, argparse, glob
from collections import defaultdict

def getTitle(url):
    r = requests.get(url)
    soup = bs(r.content, 'lxml')
    return soup.select_one('title').text

parser = argparse.ArgumentParser('Common Web Data Preprocessing')
parser.add_argument('--dir', type=str)
parser.add_argument('--nrows', type=int, default=100)
args = parser.parse_args()

files = []
for file in glob.glob("%s/*.gz" % args.dir):
    files.append(file)

for file in files:
    with open(file, 'rb') as fd:
        gzip_fd = gzip.GzipFile(fileobj=fd)
        df = pd.read_csv(gzip_fd, nrows=args.nrows, names=['text'], error_bad_lines=False)

    memo = defaultdict(set)
    for idx, row in df.iterrows():

        l = row.text.split()
        try:
            if "schema.org" in l[1]:
                website = l[-2].replace("<","").replace(">","")
                if "https://www" not in website:
                    continue
                p = l[1].split("/")[-1].replace(">","")
                memo[website].add(p)
        except:
            continue

    for website in memo:
        try:
            title = getTitle(website)
            if detect(title) == "en":
                with open(file.replace(".gz", ".tsv"), 'a') as the_file:
                    for p in memo[website]:
                        the_file.write('%s\t%s\n' % (title, p))
        except:
            continue
