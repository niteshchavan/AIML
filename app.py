from langchain_community.chat_message_histories import SQLChatMessageHistory
from flask import Flask, request, render_template, jsonify
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize chat message history
session_id = "1"
connection_string = "sqlite:///sqlite.db"

chat_message_history = SQLChatMessageHistory(
    session_id=session_id, connection_string=connection_string
)

# Define Model
llm = ChatOllama(model="qwen2:0.5b")

# Define Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Your name is Jarvis"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{context}\n\nQ: {question}\nA:"),
    ]
)

# Create Chain with History
chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string=connection_string
    ),
    input_messages_key="question",
    history_messages_key="history",
)

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

chroma_db = 'chromadb'

embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat3.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query_text = data.get("query_text", "")
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        logger.info(f"Received query: {query_text}")
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 2,
                    "score_threshold": 0.1,
                },
        )
        relevant_documents = retriever.invoke(query_text)
        results = "\n\n".join([doc.page_content for doc in relevant_documents])
        # Invoke chain with history
        config = {"configurable": {"session_id": session_id}}
        response = chain_with_history.invoke({"question": query_text, "context": results}, config=config)
        
        # Return the AI response
        return jsonify({'message': response.content}), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/geturl', methods=['POST'])
def geturl():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'message': 'No URL provided'}), 400
        
    loader = RecursiveUrlLoader(url, extractor=bs4_extractor)

    # Split the cleaned documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    pages = loader.load_and_split(text_splitter=text_splitter)

    Chroma.from_documents(pages, embedding_function, persist_directory=chroma_db)

    return jsonify({'message': f'Url {url} uploaded successfully'})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
