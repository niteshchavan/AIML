import os
from flask import Flask, request, render_template, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
chroma_db = 'chromadb'

embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

#Define Model
llm = ChatOllama(model="qwen2:0.5b")

#Memory Management
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")
runnable_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

# Create Prompt Template
prompt_template = ChatPromptTemplate.from_messages([("system", "You are a bot your name is Alice you should reply in 100 words or less"), ("human", "{context}\n\nQ: {query}\nA:")] )


@app.route('/')
def index():
    return render_template('index2.html')



@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400    
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)    
    loader = PyMuPDFLoader(filepath)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    pages = loader.load_and_split()
    chunks = text_splitter.split_documents(pages)
    Chroma.from_documents(chunks, embedding_function, persist_directory=chroma_db)
    print("File Processed.")
    return jsonify({'message': f'File {filename} uploaded successfully'})



@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query_text", "")
    retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 1,
                "score_threshold": 0.1,
            },
        )
    relevant_documents = retriever.invoke(query_text)
    results = "\n\n".join([doc.page_content for doc in relevant_documents])
    formatted_prompt = prompt_template.format(context=results, query=query_text)
        # Create a properly structured message dictionary
    message = {
        "role": "user",
        "content": formatted_prompt
    }
    llm_response = runnable_with_history.invoke([HumanMessage(content=formatted_prompt)],config={"configurable": {"session_id": "1"}},)
    print(llm_response.content)
    return jsonify({'results': llm_response.content}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)