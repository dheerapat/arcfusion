import json
import typer
from rich import print
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_community.tools import BraveSearch
from langchain_core.messages import BaseMessage
from typing import List
from llm import LLMProcessor
from vector_store.vector_store import PDFVectorStore


class GraphState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    generation: Optional[str]
    keyword: str
    web_search: Optional[str]
    documents: List[str]


### Nodes
def routing_conversation(state):
    print("---ROUTING CONVERSATION---")
    router = LLMProcessor()
    question = state["question"]
    chat = state["chat_history"]
    return router.route_conversation(chat, question)


def generate_keyword(state):
    print("---GENERATE KEYWORD---")
    extractor = LLMProcessor()
    question = state["question"]

    keyword = extractor.extract_keyword(question)
    return {"keyword": keyword}


def retriever(state):
    print("---RETRIEVE DOCUMENTS---")
    vector_store = PDFVectorStore()
    keyword = state["keyword"]

    results = vector_store.retrieve_doc(question=keyword)
    documents = state["documents"]

    for doc in results:
        documents.append(doc.page_content)

    return {"documents": documents}


def review_documents(state):
    print("---REVIEW DOCUMENTS---")
    reviewer = LLMProcessor()
    question = state["question"]
    documents = state["documents"]

    if len(documents) == 0:
        return "not_relevant"

    return reviewer.review_documents(documents, question)


def web_search(state):
    print("---WEB SEARCH---")
    keyword = state["keyword"]
    search = BraveSearch()

    results = json.loads(search.run(keyword))
    documents = state["documents"]

    for result in results:
        documents.append(result["snippet"])

    return {"documents": documents}


def generation(state):
    print("---RESEARCH GENERATION---")
    processor = LLMProcessor()

    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    generation = processor.generate_answer(documents, question, chat_history)
    return {"generation": generation}


def create_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_keyword", generate_keyword)
    workflow.add_node("retriever", retriever)
    workflow.add_node("review_documents", review_documents)
    workflow.add_node("generation", generation)
    workflow.add_node("web_search", web_search)

    workflow.set_conditional_entry_point(
        routing_conversation, {"research": "generate_keyword", "generation": "generation"}
    )
    workflow.add_edge("generate_keyword", "retriever")
    workflow.add_conditional_edges(
        "retriever",
        review_documents,
        {"relevant": "generation", "not_relevant": "web_search"},
    )
    workflow.add_edge("web_search", "generation")

    graph = workflow.compile()
    return graph

def main(user_input: str):
    print(f"[bold green]User:[/bold green] {user_input}")
    inputs: GraphState = {
        "question": user_input,
        "chat_history": [],
        "generation": None,
        "keyword": "",
        "web_search": None,
        "documents": [],
    }
    
    result = create_workflow().invoke(inputs, stream_mode="values")
    print(f"[bold green]Assistance:[/bold green] {result["generation"]}")

if __name__ == "__main__":
    typer.run(main)