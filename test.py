from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import sys
import pandas as pd
from gquestions import initBrowser, newSearch, crawlQuestions, prettyOutputName, flatten_csv
from time import sleep
import re


def crawl(keyword):
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
        # print(start_paa)

        # if len(start_paa) > 0:
        _path = 'csv/' + prettyOutputName(query, 'txt')
        with open(_path, 'w') as f:
            for item in start_paa:
                f.write("%s\n" % item.text)

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

    def __init__(self, base_url):

        self.base_url = base_url
        self.root_url = '{}://{}'.format(urlparse(self.base_url).scheme, urlparse(self.base_url).netloc)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.scraped_pages = set([])
        self.to_crawl = Queue()
        regex = re.compile('[^a-zA-Z]')
        df = pd.read_csv("data/articles.txt", error_bad_lines=False).values.tolist()
        unique_set = set([])
        for i in df:
            _ = regex.sub(' ', i[0])
            unique_set.add(_)
        for i in unique_set:
            self.to_crawl.put(i)
        # self.to_crawl.put("how to cook pasta")



    def parse_links(self, html):
        # soup = BeautifulSoup(html, 'html.parser')
        # links = soup.find_all('a', href=True)
        # for link in links:
        #     url = link['href']
        #     if url.startswith('/') or url.startswith(self.root_url):
        #         url = urljoin(self.root_url, url)
        #         if url not in self.scraped_pages:
        self.to_crawl.put("www.google.com")

    def scrape_info(self, html):
        return

    def post_scrape_callback(self, res):
        result = res.result()
        if result and result.status_code == 200:
            self.parse_links(result.text)
            self.scrape_info(result.text)

    def scrape_page(self, url):
        try:
            crawl(url)
            # sleep(5)

        except Exception as e:
            # add to file and add to the pool
            # with open("error.txt", 'a+') as f:
            #     f.write("%s\n" % url)
            print("error: %s" % url)
            self.to_crawl.put(url)
        return

    def run_scraper(self):
        while True:
            try:
                target_url = self.to_crawl.get(timeout=10)
                # sleep(4)
                print(self.to_crawl.qsize())
                break
                # if target_url not in self.scraped_pages:
                    # self.scraped_pages.add(target_url)
                    # self.pool.submit(self.scrape_page, target_url)

            except Empty:
                break

            except Exception as e:
                print(e)
                print("main error: %s" % target_url)
                continue
                # break
if __name__ == '__main__':
    # s = MultiThreadScraper("http://www.google.co.uk")
    # s.run_scraper()
    # run on 15:37

    #
    # import time
    # start = time.time()
    # crawl("jaramana")
    # end = time.time()
    # print(end - start)

    regex = re.compile('[^a-zA-Z]')
    df = pd.read_csv("data/articles.txt", error_bad_lines=False).values.tolist()
    unique_set = set([])
    for i in df:
        _ = regex.sub(' ', i[0])
        unique_set.add(_)
    toCrawl = list(unique_set)
    # toCrawl = ["how to cook pasta"]
    # print(crawl("how to cook pasta"))

    count = 0
    # TODO skip already crawled articles
    # for i in range(len(toCrawl)):
    #     print(i, toCrawl[i])
    #     res = crawl(toCrawl[i])
    #
    #     if len(res) == 0:
    #         count += 1
    #     else:
    #         count = 0
    #
    #     if count == 5:
    #         break


