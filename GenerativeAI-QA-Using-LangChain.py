import gradio as gr
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory  # Using the updated memory package
from langchain_community.vectorstores import Chroma  # Corrected import for Chroma
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAIEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import StructuredTool
from langchain.callbacks.base import BaseCallbackHandler

# ================================
# Step 1: Setup Logging for Debugging
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# Step 2: Load Wikipedia Data
# ================================
def fetch_wikipedia_content():
    """Fetches Wikipedia content using LangChain's WikipediaLoader."""
    loader = WikipediaLoader(query="Generative artificial intelligence", lang="en")
    documents = loader.load()
    return documents[0].page_content if documents else "Page not found."

wiki_text = fetch_wikipedia_content()

# ================================
# Step 3: Process Wikipedia Text for Retrieval
# ================================
def process_and_store_wikipedia(text):
    """Splits Wikipedia content into chunks, embeds them, and stores in ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()  # Using updated OpenAI embeddings
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="/home/user/chroma_db")  # Ensuring persistence
    return vectorstore.as_retriever()

retriever = process_and_store_wikipedia(wiki_text)

# ================================
# Step 4: Initialize Chat Model and Memory
# ================================
llm = ChatOpenAI(model_name="gpt-4o")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Initialize memory for conversation history

# ================================
# Step 5: Create Q/A Retrieval Chain
# ================================
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory
)

# ================================
# Step 6: Implement Chatbot Response Function with Caching
# ================================
def ask_with_memory(query):
    """Retrieves the answer from memory if available, otherwise fetches it using LangChain's Q/A chain."""

    # Load chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Check if the exact query has been answered before
    for i in range(len(chat_history) - 1):
        if chat_history[i].content == query:
            return chat_history[i + 1].content  # Return cached answer

    # If not cached, process the query
    response = qa_chain.invoke({"question": query})["answer"]

    # Save query-response pair in memory
    memory.save_context({"question": query}, {"answer": response})

    return response


# ================================
# Step 7: Implement Structured Function Calling for Section Extraction
# ================================
def extract_section_by_query(query: str) -> str:
    """Finds and returns the most relevant section based on a user query using embeddings."""
    vector_store = retriever  # Use the existing retriever

    # Retrieve the most relevant section
    retrieved_docs = vector_store.get_relevant_documents(query)

    if not retrieved_docs:
        return "Section not found."

    return f"Section: {retrieved_docs[0].metadata.get('title', 'Unknown')}\n\n{retrieved_docs[0].page_content}"

section_extraction_tool = StructuredTool.from_function(
    extract_section_by_query,
    name="extract_section_by_query",
    description="Finds the most relevant Wikipedia section based on a user query using embeddings."
)

# ================================
# Step 8: Implement Callback Logging for Debugging
# ================================
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        logger.info(f"Starting chain execution with input: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        logger.info(f"Chain execution finished. Output: {outputs}")

callback_handler = LoggingCallbackHandler()
qa_chain.callbacks = [callback_handler]

# ================================
# Step 9: Define Gradio Interface
# ================================
def respond(message, history, system_message, max_tokens, temperature, top_p):
    """
    Processes user query and retrieves answers from Wikipedia-based Q/A system with caching.
    """
    return ask_with_memory(message)

# ================================
# Step 10: Create Gradio Interface
# ================================
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are an AI expert answering questions about Generative AI.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

if __name__ == "__main__":
    demo.launch()