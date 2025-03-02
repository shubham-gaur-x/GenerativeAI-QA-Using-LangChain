# Generative AI Q/A System Using LangChain

## Overview
This repository contains a Question-Answering (Q/A) system that leverages LangChain Core to retrieve and answer user queries based on the content of the Wikipedia page on Generative Artificial Intelligence. The solution incorporates data retrieval, caching, structured function calling, and callback handling to ensure efficient and accurate responses.

The application is deployed on Hugging Face Spaces and can be accessed [here](https://huggingface.co/spaces/ShubhamGaur/GenerativeAI-QA-Using-LangChain).

## Features
- **Data Retrieval:** Fetches content from Wikipedia using LangChain's `WikipediaLoader`.
- **Q/A Processing:** Builds a conversational Q/A system using LangChain’s `ConversationalRetrievalChain`.
- **Caching Mechanism:** Implements `ConversationBufferMemory` to cache previously answered queries.
- **Function Calling:** Extracts relevant sections from the Wikipedia content based on user queries.
- **Callback Handling:** Implements a custom `LoggingCallbackHandler` to track and log events during the execution.
- **Gradio Interface:** Provides an easy-to-use web interface for interacting with the Q/A system.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.8 or higher installed. Then, install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**

   Launch the Gradio interface to interact with the Q/A system:

   ```bash
   python app.py
   ```

   The application will run locally on `http://0.0.0.0:7860`. To create a public link, set `share=True` in `launch()`.

## Components

### 1. **Data Retrieval:**
   - **WikipediaLoader:** Fetches the content from the Wikipedia page on "Generative Artificial Intelligence."
   - **Code Implementation:**
     ```python
     def fetch_wikipedia_content():
         loader = WikipediaLoader(query="Generative artificial intelligence", lang="en")
         documents = loader.load()
         return documents[0].page_content if documents else "Page not found."
     ```

### 2. **Q/A Processing:**
   - **ConversationalRetrievalChain:** Combines the LLM (ChatGPT) with the retriever to process the user’s query and fetch relevant answers.
   - **Code Implementation:**
     ```python
     qa_chain = ConversationalRetrievalChain.from_llm(
         llm, retriever=retriever, memory=memory
     )
     ```

### 3. **Caching Mechanism:**
   - **ConversationBufferMemory:** Stores query-answer pairs to avoid redundant retrievals.
   - **Code Implementation:**
     ```python
     def ask_with_memory(query):
         chat_history = memory.load_memory_variables({})["chat_history"]
         for i in range(len(chat_history) - 1):
             if chat_history[i].content == query:
                 return chat_history[i + 1].content  # Return cached answer
         response = qa_chain.invoke({"question": query})["answer"]
         memory.save_context({"question": query}, {"answer": response})
         return response
     ```

### 4. **Function Calling:**
   - **StructuredTool:** Extracts specific sections from the Wikipedia content based on the query.
   - **Code Implementation:**
     ```python
     def extract_section_by_query(query: str) -> str:
         vector_store = retriever
         retrieved_docs = vector_store.get_relevant_documents(query)
         if not retrieved_docs:
             return "Section not found."
         return f"Section: {retrieved_docs[0].metadata.get('title', 'Unknown')}

{retrieved_docs[0].page_content}"
     ```

### 5. **Callback Handling:**
   - **LoggingCallbackHandler:** Logs the execution flow of the Q/A chain for debugging and monitoring.
   - **Code Implementation:**
     ```python
     class LoggingCallbackHandler(BaseCallbackHandler):
         def on_chain_start(self, serialized, inputs, **kwargs):
             logger.info(f"Starting chain execution with input: {inputs}")
         def on_chain_end(self, outputs, **kwargs):
             logger.info(f"Chain execution finished. Output: {outputs}")
     
     callback_handler = LoggingCallbackHandler()
     qa_chain.callbacks = [callback_handler]
     ```

### 6. **Gradio Interface:**
   - The Gradio interface enables users to input queries and receive answers from the Q/A system. It includes settings like `temperature`, `max tokens`, and `top-p` to customize the model’s behavior.
   - **Code Implementation:**
     ```python
     demo = gr.ChatInterface(
         respond,
         additional_inputs=[
             gr.Textbox(value="You are an AI expert answering questions about Generative AI.", label="System message"),
             gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
             gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
             gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
         ],
     )
     ```

## Example Queries

1. **Applications of Generative AI:**  
   Query: "What are the applications of Generative AI?"  
   Output: "Generative AI has applications across a wide range of industries, including Software Development, Healthcare, Finance, etc."

2. **Limitations of Generative AI:**  
   Query: "What are the limitations of Generative AI?"  
   Output: "Generative AI has several limitations, including Quality and Accuracy, Bias and Fairness, Misuse and Ethical Concerns, etc."

3. **Models used in Generative AI:**  
   Query: "What are the models used in Generative AI?"  
   Output: "Common models used in Generative AI include large language models (LLMs) such as ChatGPT, Copilot, Gemini, and LLaMA, etc."

## Conclusion
This Q/A system effectively answers user queries based on the information from the Wikipedia page on Generative Artificial Intelligence, integrating data retrieval, caching, function calling, and callback handling to ensure efficiency and accurate results.

