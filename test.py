from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import sys
import pandas as pd
from gquestions import initBrowser, newSearch, crawlQuestions, prettyOutputName, flatten_csv


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

        initialSet = {}
        cnt = 0
        for q in start_paa:
            initialSet.update({cnt: q})
            cnt += 1

        paa_list = []

        crawlQuestions(start_paa, paa_list, initialSet, query, browser, depth)
        # treeData = 'var treeData = ' + json.dumps(paa_list) + ';'
        # if paa_list[0]['children']:
        # root = os.path.dirname(os.path.abspath(__file__))
        # templates_dir = os.path.join(root, 'templates')
        # env = Environment(loader=FileSystemLoader(templates_dir))
        # template = env.get_template('index.html')
        # filename = os.path.join(root, 'html', prettyOutputName())
        # with open(filename, 'w') as fh:
        #     fh.write(template.render(
        #         treeData=treeData,
        #     ))

    if args['--csv']:
        if paa_list[0]['children']:
            _path = 'tmp/' + prettyOutputName(query, 'csv')
            flatten_csv(paa_list, depth, _path)

    browser.close()

class MultiThreadScraper:

    def __init__(self, base_url):

        self.base_url = base_url
        self.root_url = '{}://{}'.format(urlparse(self.base_url).scheme, urlparse(self.base_url).netloc)
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.scraped_pages = set([])
        self.to_crawl = Queue()
        df = pd.read_csv("data/articles.txt", error_bad_lines=False).values.tolist()[:10]
        for i in df:
            self.to_crawl.put(i[0])

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
        except Exception as e:
            print("Scrape page")
            print(e)
            return

    def run_scraper(self):
        while True:
            try:
                target_url = self.to_crawl.get(timeout=10)
                if target_url not in self.scraped_pages:
                    # print(target_url)
                    self.scraped_pages.add(target_url)
                    self.pool.submit(self.scrape_page, target_url)
                    # job.add_done_callback(self.post_scrape_callback)
            except Empty as e:
                print("Empty")
                print(target_url)
                return
            except Exception as e:
                print("run scrapper")
                print(e)
                continue
if __name__ == '__main__':
    # start = time.time()
    s = MultiThreadScraper("http://www.google.co.uk")
    s.run_scraper()
    # end = time.time()
    # print(end - start)