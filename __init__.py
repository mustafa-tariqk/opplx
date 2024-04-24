import sys
import urllib3
from googlesearch import search

NUM_RESULTS = 5
QUESTION = ' '.join(sys.argv[1:])
http = urllib3.PoolManager()

links = search(QUESTION, num_results=NUM_RESULTS, lang="en")
link_response_dict = {link: http.request('GET', "https://r.jina.ai/" + link) for link in links}