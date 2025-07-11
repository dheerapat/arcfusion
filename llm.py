from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Literal

load_dotenv()


class ConversationRoute(BaseModel):
    route: Literal["generation", "research"]


class Keyword(BaseModel):
    search_keyword: str


class DocumentRelevancy(BaseModel):
    relevancy: Literal["relevant", "not_relevant"]


class LLMProcessor:
    def __init__(self, model_name: str = "gpt-4.1-nano", temperature: float = 0.2):
        self.llm = init_chat_model(
            model_name, model_provider="openai", temperature=temperature
        )

        # Create structured LLM instances for different output types
        self.route_llm = self.llm.with_structured_output(ConversationRoute)
        self.keyword_llm = self.llm.with_structured_output(Keyword)
        self.relevancy_llm = self.llm.with_structured_output(DocumentRelevancy)

        # System prompts for different tasks
        self.prompts = {
            "routing": """
            You are an expert conversation router that determines the optimal processing path for user queries.
            
            Your task is to analyze the conversation context and current user input to decide between:
            - 'generation': For requests that can be answered directly with existing knowledge, creative tasks, explanations, or general assistance
            - 'research': For requests requiring current information, specific facts, data retrieval, or document-based answers
            
            Consider these factors:
            1. Does the query ask for recent/current information?
            2. Does it require specific facts or data that might need verification?
            3. Is it a general knowledge question that can be answered directly?
            4. Does it involve creative or analytical tasks?
            
            Route to 'research' if the query involves:
            - Current events, news, or time-sensitive information
            - Specific statistics, data, or factual claims that need verification
            - Questions about documents, studies, or sources
            - Requests for up-to-date information
            
            Route to 'generation' if the query involves:
            - General explanations or educational content
            - Creative writing or brainstorming
            - Analysis or opinion-based responses
            - How-to guides or tutorials
            - General conversation or advice
            """,
            "keyword_extraction": """
            You are an expert at extracting optimal search keywords from user queries for information retrieval systems.
            
            Your goal is to identify the most effective search terms that will retrieve relevant documents from vector stores or search engines.
            
            Guidelines for keyword extraction:
            1. Focus on the core subject matter and key concepts
            2. Include specific terms, names, or technical vocabulary
            3. Remove filler words (the, and, or, but, how, what, when, where, why)
            4. Preserve important qualifiers (recent, best, top, latest, specific dates/years)
            5. Consider synonyms and related terms that might appear in relevant documents
            6. Keep it concise but comprehensive (2-6 key terms typically optimal)
            
            Examples:
            - "What is the latest research on climate change effects?" → "climate change research effects latest"
            - "How do I cook pasta properly?" → "cook pasta properly technique"
            - "Tell me about machine learning algorithms for beginners" → "machine learning algorithms beginners"
            - "What are the benefits of exercise for mental health?" → "exercise benefits mental health"
            
            Extract the most relevant search keywords that will help find documents containing the information needed to answer the user's question.
            """,
            "document_review": """
            You are an expert document relevance assessor tasked with determining whether provided documents contain sufficient information to answer a user's question.
            
            Your role is to:
            1. Carefully analyze the user's question to understand what information they seek
            2. Review all provided documents for content that addresses their query
            3. Determine if the documents collectively provide enough relevant information
            
            Assessment criteria:
            - RELEVANT: Documents contain information that directly answers the question, provides necessary context, or offers useful insights related to the query
            - NOT_RELEVANT: Documents lack the information needed to answer the question, are off-topic, or only tangentially related
            
            Consider these factors:
            - Does the content directly address the user's question?
            - Is there sufficient detail to provide a meaningful answer?
            - Are key concepts, facts, or data present in the documents?
            - Would these documents help someone understand the topic the user is asking about?
            
            Be thorough but decisive. If documents contain even partial relevant information that could contribute to answering the question, lean toward "relevant."
            
            Available Documents:
            {documents_text}
            
            Analyze whether these documents are relevant to answering the user's question.
            """,
        }

    def route_conversation(
        self, existing_conversation: list[BaseMessage], user_input: str
    ) -> Optional[str]:
        """Route conversation based on existing context and user input"""
        try:
            conversation = [
                SystemMessage(content=self.prompts["routing"]),
                *existing_conversation,  # Spread the existing conversation
                HumanMessage(content=user_input),
            ]
            result = self.route_llm.invoke(conversation)
            return getattr(result, "route", None)
        except Exception as e:
            print(f"Error routing conversation: {e}")
            return None

    def extract_keyword(self, user_input: str) -> Optional[str]:
        """Extract search keywords from user input"""
        try:
            conversation = [
                SystemMessage(content=self.prompts["keyword_extraction"]),
                HumanMessage(content=user_input),
            ]
            result = self.keyword_llm.invoke(conversation)
            return getattr(result, "search_keyword", None)
        except Exception as e:
            print(f"Error extracting keyword: {e}")
            return None

    def review_documents(self, docs: list[Document], user_input: str) -> Optional[str]:
        """Review documents for relevancy to user question"""
        try:
            # Create documents text for the prompt
            documents_text = ""
            for i, doc in enumerate(docs, 1):
                documents_text += f"Document {i}:\n{doc.page_content}\n\n"

            system_prompt = self.prompts["document_review"].format(
                documents_text=documents_text
            )

            conversation = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User question: {user_input}"),
            ]

            result = self.relevancy_llm.invoke(conversation)
            return getattr(result, "relevancy", None)
        except Exception as e:
            print(f"Error reviewing documents: {e}")
            return None


if __name__ == "__main__":
    processor = LLMProcessor()

    test_inputs = [
        "What is the name of president of USA",
        "How to cook pasta",
        "What are the benefits of exercise?",
        "Tell me about machine learning algorithms",
    ]

    print("Testing Keyword Extraction:")
    print("-" * 50)

    for user_input in test_inputs:
        keyword = processor.extract_keyword(user_input)
        print(f"Input: {user_input}")
        print(f"Keyword: {keyword}")
        print()

    print("Testing Document Review:")
    print("-" * 50)

    sample_docs = [
        Document(
            page_content="The current president of the United States of America is Donald J. Trump, He was elected into the office in 2024"
        ),
        Document(page_content="President elections in the US occur every four years"),
    ]

    user_question = "What is the name of president of USA"

    relevancy = processor.review_documents(sample_docs, user_question)
    print(f"Question: {user_question}")
    print(f"Document relevancy: {relevancy}")

    print("\nTesting Conversation Routing:")
    print("-" * 50)

    existing_conversation = [
        HumanMessage(content="Hello, I need help with something"),
        AIMessage(content="Hello! I'm here to help. What do you need assistance with?"),
    ]

    route = processor.route_conversation(
        existing_conversation, "I want to research about AI"
    )
    print(f"User: I want to research about AI")
    print(f"Route: {route}")
