import os
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
chroma_db = "chroma_dbs"
db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)
llm = Ollama(model="qwen2:0.5b")




@app.route('/')
def index():
    return render_template('index.html')

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
    return jsonify({'message': f'File {filename} uploaded successfully'})


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query_text", "")
    print(query_text)
    docs = db.similarity_search(query_text, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    #context = "todays date is 24 july 2024"
    print(context)
    #results = [doc.page_content for doc in docs]
    prompt_template = ChatPromptTemplate.from_messages([("system", "You are a bot you should reply in 100 words or less"), ("human", "{context}\n\nQ: {query}\nA:")] )
    formatted_prompt = prompt_template.format(context=context, query=query_text)
    #results = llm(formatted_prompt)
    # Create a properly structured message dictionary
    message = {
        "role": "user",
        "content": formatted_prompt
    }
    
    results = llm.invoke([message])
    print(results)
    return jsonify({'results': results}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)