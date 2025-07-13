from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Literal, List

load_dotenv()


class ConversationRoute(BaseModel):
    route: Literal["generation", "research"]


class Keyword(BaseModel):
    search_keyword: str


class DocumentRelevancy(BaseModel):
    relevancy: Literal["relevant", "not_relevant"]


class LLMProcessor:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.2):
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
            You are an intelligent conversation router for a comprehensive information system.
            Your primary goal is to determine if a user's query requires fetching new or specific external information (research) or can be answered using general knowledge, creative abilities, or by summarizing existing context (generation).

            You are equipped with a powerful data retrieval system and web search capabilities.
            **Prioritize 'research' if the user's question explicitly or implicitly requests specific, factual, up-to-date, or external information that requires lookup.**
            **Do NOT default to 'generation' if the answer needs to be verified or sourced externally.**

            Your task is to analyze the conversation context and current user input to decide between:

            -   **'research'**: Choose 'research' if the query involves:
                * **Specific factual questions**: e.g., "What is the capital of France?", "Who invented the internet?", "When was the last election?".
                * **Requests for detailed information on specific topics**: e.g., "Explain quantum entanglement in detail", "What are the different types of AI agents?".
                * **Current events, news, or time-sensitive data**: e.g., "What's the latest news on climate change?", "Who is the current CEO of Company X?".
                * **Statistics, data, or verifiable claims**: e.g., "What is the population of Tokyo?", "Tell me about the average lifespan of a golden retriever.".
                * **Information from specific documents, studies, or sources**: If the user is asking about content that would likely be in a document or database.
                * **Comparative or analytical questions requiring specific data**: e.g., "Compare the economic policies of country A and country B".

            -   **'generation'**: Choose 'generation' if the query involves:
                * **General explanations or educational content that do NOT require specific external lookup**: e.g., "How does a computer work (generally)?", "What are common ways to relax?".
                * **Creative tasks**: e.g., "Write me a short story about a brave knight", "Suggest ideas for a birthday party.".
                * **Opinion-based responses or general advice**: e.g., "What's your opinion on remote work?", "How can I stay motivated?".
                * **Summarizing or rephrasing information already present in the chat history**.
                * **Conversational greetings, small talk, or general assistance that doesn't need external facts**: e.g., "Hello", "How are you?", "Can you rephrase that?".
                * **Broad, conceptual questions where an extensive, factual deep-dive is not implied.**

            **Decision Logic:**
            1.  Read the user's question carefully.
            2.  Ask yourself: "Can I answer this accurately and comprehensively without needing to search for specific, verifiable facts or external documents?"
            3.  If the answer is "No" (meaning external lookup is required or highly beneficial), route to 'research'.
            4.  If the answer is "Yes" (meaning it's general knowledge, creative, or conversational), route to 'generation'.

            Think step by step and justify your decision before outputting the route.
            """,
            "keyword_extraction": """
            You are an expert at extracting optimal search keywords from user queries for information retrieval systems.
            
            Your goal is to identify the most effective search terms that will retrieve relevant documents from vector stores or search engines.
            
            Guidelines for keyword extraction:
            1. Focus on the core subject matter and key concepts.
            2. Include specific terms, names, or technical vocabulary.
            3. Remove filler words (the, and, or, but, how, what, when, where, why).
            4. Preserve important qualifiers (recent, best, top, latest, specific dates/years).
            5. Consider synonyms and related terms that might appear in relevant documents.
            6. Keep it concise but comprehensive (2-6 key terms typically optimal).
            
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
            1. Carefully analyze the user's question to understand what information they seek.
            2. Review all provided documents for content that addresses their query.
            3. Determine if the documents collectively provide enough relevant information.
            
            Assessment criteria:
            - 'relevant': Documents contain information that directly answers the question, provides necessary context, or offers useful insights related to the query.
            - 'not_relevant': Documents lack the information needed to answer the question, are off-topic, or only tangentially related.
            
            Consider these factors:
            - Does the content directly address the user's question?
            - Is there sufficient detail to provide a meaningful answer?
            - Are key concepts, facts, or data present in the documents?
            - Would these documents help someone understand the topic the user is asking about?
            
            Be thorough but decisive. If documents contain information that could contribute to answering the question, lean toward "relevant."
            
            Available Documents:
            {documents_text}
            
            Analyze whether these documents are relevant to answering the user's question.
            """,
            "answer_generation": """
            You are an expert research assistant that provides comprehensive, accurate, and well-structured answers based on provided documents and context.
            
            Your task is to:
            1. Analyze the user's question thoroughly.
            2. Review all provided documents for relevant information.
            3. Synthesize the information into a coherent, informative response.
            4. Ensure accuracy and avoid speculation beyond what's supported by the documents.
            
            Guidelines for generating answers:
            - Start with a direct answer to the user's question when possible.
            - Use information from the documents to support your response.
            - Organize information logically and clearly.
            - Cite or reference key information when relevant.
            - If documents contain conflicting information, acknowledge this.
            - If the question cannot be fully answered from the documents, state what information is available.
            - Maintain an informative but conversational tone.
            - Be comprehensive but concise.
            
            Available Context Documents:
            {context_documents}
            
            Based on the provided documents, answer the user's question thoroughly and accurately.
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
                documents_text += f"Document {i}:\n{doc}\n\n"

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

    def generate_answer(
        self,
        documents: List[str],
        user_question: str,
        chat_history: List[BaseMessage] = [],
    ):
        try:
            context_documents = ""
            for i, doc in enumerate(documents, 1):
                context_documents += f"Document {i}:\n{doc.strip()}\n\n"

            system_prompt = self.prompts["answer_generation"].format(
                context_documents=context_documents
            )

            conversation: List[BaseMessage] = [SystemMessage(content=system_prompt)]

            if chat_history:
                conversation.extend(chat_history)

            conversation.append(HumanMessage(content=user_question))

            response = self.llm.invoke(conversation)

            if hasattr(response, "content"):
                return response.content
            else:
                return str(response)

        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I encountered an error while generating the answer: {str(e)}"


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
