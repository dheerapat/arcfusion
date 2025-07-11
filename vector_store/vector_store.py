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
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.6},
        )

        for _, v in enumerate(urls):
            self.insert_doc(v)

    def insert_doc(self, url: str):
        docs = WebBaseLoader(url).load()
        doc_splits = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(doc_splits)

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
