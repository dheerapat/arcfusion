import operator
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing import List, Annotated
from llm import LLMProcessor
from vector_store.vector_store import URLVectorStore


class GraphState(TypedDict):
    question: str
    max_retries: int
    loop_step: Annotated[int, operator.add]
    chat_history: List[BaseMessage]
    generation: Optional[str]
    keyword: Optional[str]
    web_search: Optional[str]
    answers: Optional[int]
    documents: Optional[List[Document]]


### Nodes
def routing_conversation(state):
    print("---ROUTING CONVERSATION---")
    router = LLMProcessor()
    question = state["question"]
    chat = state["chat_history"]
    print(router.route_conversation(chat, question))


def generate_keyword(state):
    print("---GENERATE KEYWORD---")
    extractor = LLMProcessor()
    question = state["question"]

    keyword = extractor.extract_keyword(question)
    return {"keyword": keyword}


def retriever(state):
    print("---RETRIEVE DOCUMENTS---")
    vector_store = URLVectorStore()
    keyword = state["keyword"]

    documents = vector_store.retrieve_doc(keyword)
    return {"documents": documents}


def review_documents(state):
    print("---REVIEW DOCUMENTS---")
    reviewer = LLMProcessor()
    question = state["question"]
    documents = state["documents"]
    if len(documents) == 0:
        print("not_relevant")

    print(reviewer.review_documents(documents, question))


workflow = StateGraph(GraphState)

workflow.add_node("routing_conversation", routing_conversation)
workflow.add_node("generate_keyword", generate_keyword)
workflow.add_node("retriever", retriever)
workflow.add_node("review_documents", review_documents)

workflow.set_entry_point("routing_conversation")
# workflow.add_edge("generate_keyword", "retriever")
# workflow.add_edge("retriever", "review_documents")

graph = workflow.compile()

inputs: GraphState = {
    "question": "How many example are enough for good accuracy?",
    "max_retries": 3,
    "loop_step": 0,
    "chat_history": [],
    "generation": None,
    "keyword": None,
    "web_search": None,
    "answers": None,
    "documents": None,
}

for event in graph.stream(inputs, stream_mode="values"):
    print(event)
