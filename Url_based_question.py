import re
from flask import Flask, request, render_template, jsonify
from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


app = Flask(__name__)

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


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query_text", "")
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
            search_kwargs={
                "k": 2,
                "score_threshold": 0.1,
            },
    )
    print(retriever)
    relevant_documents = retriever.invoke(query_text)
    
    results = "\n\n".join([doc.page_content for doc in relevant_documents])
    # Combine page content with metadata into a formatted string
    #results = "\n\n".join(
    #    f"Page {doc.metadata.get('page', 'Unknown')}:\n{doc.page_content}"
    #    for doc in relevant_documents
    #)
    #print(results)
    formatted_prompt = prompt_template.format(context=results, query=query_text)
    
    print(formatted_prompt)
    llm_response = runnable_with_history.invoke([HumanMessage(content=formatted_prompt)],config={"configurable": {"session_id": "1"}},)
    #print(llm_response)
    return jsonify({'results': llm_response.content}), 200
    #llm_response = 'test'
    #print(llm_response)
    #print(formatted_prompt)
    #return jsonify({'results': results}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)