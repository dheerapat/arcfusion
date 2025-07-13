from fastapi import FastAPI
from pydantic import BaseModel
from graph import GraphState, create_workflow

class Question(BaseModel):
    question: str

app = FastAPI()


@app.get("/", status_code=204)
async def root():
    return

@app.post("/ask")
async def ask(question: Question):
    graph = create_workflow()
    inputs: GraphState = {
        "question": question.question,
        "chat_history": [],
        "generation": None,
        "keyword": "",
        "web_search": None,
        "documents": [],
    }
    result = graph.invoke(input=inputs)
    return result["generation"]