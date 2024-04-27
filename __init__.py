"""
LLM + Web Search Engine
"""
import sys
from typing import List

import urllib3
from googlesearch import search
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load(question: str, top_k: int) -> List[Document]:
    """
    Load documents based on user question
    """
    print('Searching for:', question)
    http = urllib3.PoolManager()
    links = search(question, num_results=top_k, lang="en")
    documents = []
    for link in links:
        # Eventually use jina locally / some other way to get the page content
        documents.append(Document(page_content=http.request(
            'GET', "https://r.jina.ai/" + link).data, metadata={'source': link}))

    # write documents to files
    # for i, doc in enumerate(documents):
    #     with open(f"doc_{i}.txt", "w", encoding="utf-8") as f:
    #         f.write(doc.page_content)

    return documents

def load_from_files(path: str) -> List[Document]:
    """
    Create documents from files
    """
    documents = []
    for i in range(5):
        with open(f"{path}/doc_{i}.txt", "r", encoding="utf-8") as f:
            page_content = f.read()
        documents.append(Document(page_content=page_content, metadata={'source': i}))
    return documents


def process(documents: List[Document]) -> List[Document]:
    """
    Process documents for better indexing
    """
    print('Processing documents')
    text_splitter = RecursiveCharacterTextSplitter()
    split_documents = text_splitter.split_documents(documents)
    return split_documents


def retrieve(documents: List[Document], top_k: int = 5) -> VectorStoreRetriever:
    """
    Find useful information from documents
    """
    print('Retrieving information')
    embeddings_function = OllamaEmbeddings(model="llama3")
    db = FAISS.from_documents(documents, embeddings_function)
    return db.as_retriever(search_kwargs={'k': top_k})


def main() -> None:
    """
    Handle all input and output, generate response
    """
    question = ' '.join(sys.argv[1:])
    llm = ChatOllama(model="llama3")
    loads = load_from_files('.')
    processed = process(loads)
    retriever = retrieve(processed, 5)
    print('Generating answer')
    qa = RetrievalQA.from_llm(llm, retriever=retriever)
    print(qa.invoke({"query": question})["result"])


if __name__ == '__main__':
    main()
