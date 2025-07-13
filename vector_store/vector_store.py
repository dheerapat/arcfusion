import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class URLVectorStore:
    def __init__(self):
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        self.vectorstore = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        )
        self.retriever = self.vectorstore.as_retriever(kwargs={"k": 3})

        for _, v in enumerate(urls):
            self.insert_doc(v)

    def insert_doc(self, url: str):
        docs = WebBaseLoader(url).load()
        doc_splits = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(doc_splits)

    def retrieve_doc(self, question: str):
        return self.retriever.invoke(question)


class PDFVectorStore:
    def __init__(self, pdf_directory="./paper"):
        self.pdf_directory = pdf_directory
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        self.vectorstore = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        )
        self.retriever = self.vectorstore.as_retriever(kwargs={"k": 3})

        self.load_all_pdfs()

    def load_all_pdfs(self):
        """Load all PDF files from the specified directory"""
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return

        for pdf_file in pdf_files:
            print(f"Loading: {pdf_file}")
            self.insert_pdf(pdf_file)

        print(f"Total PDF files processed: {len(pdf_files)}")

    def insert_pdf(self, pdf_path: str):
        try:
            loader = PyPDFLoader(file_path=pdf_path, mode="single", pages_delimiter="")

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)

            if docs:
                doc_splits = self.text_splitter.split_documents(docs)
                self.vectorstore.add_documents(doc_splits)
                print(f"Successfully added {len(doc_splits)} chunks from {pdf_path}")
            else:
                print(f"No content loaded from {pdf_path}")

        except Exception as e:
            print(f"Error loading {pdf_path}: {str(e)}")

    def add_single_pdf(self, pdf_path: str):
        if os.path.exists(pdf_path):
            self.insert_pdf(pdf_path)
        else:
            print(f"PDF file not found: {pdf_path}")

    def retrieve_doc(self, question: str):
        return self.retriever.invoke(question)


if __name__ == "__main__":
    # urls = [
    #     "https://lilianweng.github.io/posts/2023-06-23-agent/",
    #     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    #     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    # ]
    vector_store = URLVectorStore()

    # for i, v in enumerate(urls):
    #     vector_store.insert_doc(v)

    question = "What is Chain of thought prompting?"
    docs = vector_store.retrieve_doc(question)
    print(len(docs))

    doc_txt = docs[1].page_content
    print(doc_txt)
