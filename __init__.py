import sys
import urllib3
from googlesearch import search

NUM_RESULTS = 5

def search(question, top_k=5):
    http = urllib3.PoolManager()
    links = search(question, num_results=top_k, lang="en")
    return {link: http.request('GET', "https://r.jina.ai/" + link) for link in links}

def main():
    NUM_RESULTS = 5
    QUESTION = ' '.join(sys.argv[1:])
    print(search(QUESTION, NUM_RESULTS))