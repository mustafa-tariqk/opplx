"""
LLM + Web Search Engine
"""
import sys
from typing import List

import urllib3
from googlesearch import search
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
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
            'GET', "https://r.jina.ai/" + link).data, metadata={'url': link}))
    
    # write documents to files
    for i, doc in enumerate(documents):
        with open(f"doc_{i}.txt", "w") as f:
            f.write(doc.page_content)

    return documents


def process(documents: List[Document]) -> List[Document]:
    """
    Process documents for better indexing
    """
    print('Processing documents')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    split_documents = text_splitter.split_documents(documents)
    return split_documents


def retrieve(documents: List[Document], top_k: int = 5):
    """
    Find useful information from documents
    """
    print('Retrieving information')
    embeddings_function = OllamaEmbeddings(model="llama3")
    db = FAISS.from_documents(documents, embeddings_function)
    return db.as_retriever(search_kwargs={'k': top_k})


def generate(question: str) -> str:
    """
    Generate an answer based on retrieved information
    """
    llm = ChatOllama(model="llama3")
    loads = load(question, 5)
    processed = process(loads)
    retriever = retrieve(processed, 5)
    print('Generating answer')
    qa = RetrievalQA.from_llm(llm, retriever=retriever)
    return qa.invoke({"query": question})["result"]


def main() -> None:
    """
    Handle all input and output
    """
    question = ' '.join(sys.argv[1:])
    answer = generate(question)
    print(answer)


if __name__ == '__main__':
    main()