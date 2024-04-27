"""
LLM + Web Search Engine
"""
import re
import sys

import urllib3
from googlesearch import search
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_retriever(question: str, num_links: int, num_documents: int) -> VectorStoreRetriever:
    """
    Args:
        question (str): The question to search the web for
        num_links (int): The number of links to retrieve based on question
        num_documents (int): The number of documents to retrieve based on links
    Returns:
        VectorStoreRetriever: A retriever based on the documents retrieved from the web
    """
    http = urllib3.PoolManager()
    splitter = RecursiveCharacterTextSplitter()
    embeddings_function = OllamaEmbeddings(model="llama3")

    links = search(question, num_results=num_links, lang="en")
    documents = []
    for link in links:
        documents.append(Document(page_content=http.request(
            'GET', "https://r.jina.ai/" + link).data, metadata={'source': link}))
    documents = splitter.split_documents(documents)
    db = FAISS.from_documents(documents, embeddings_function)
    return db.as_retriever(search_kwargs={'k': num_documents})


def main() -> None:
    """
    Handle all input and output, generate response
    """
    question = re.escape(' '.join(sys.argv[1:]))
    qa = RetrievalQA.from_llm(llm=ChatOllama(model="llama3"),
                              retriever=make_retriever(question, 1, 1),
                              callbacks=StreamingStdOutCallbackHandler())
    # qa.invoke({"query": question})["result"]
    chain = RunnableParallel([qa, RunnablePassthrough()]).assign(answer=question)
    for chunk in qa.stream({"query": question}):
        print(chunk)


if __name__ == '__main__':
    main()
