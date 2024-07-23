from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.chat_models import ChatOllama
from flask import Flask, request, render_template, jsonify
import os
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="qwen2:0.5b")

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

runnable_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    print(user_message)

    llm_response = runnable_with_history.invoke(
    [HumanMessage(content=user_message)],
    config={"configurable": {"session_id": "1"}},)
    
    print(llm_response.content)
    return jsonify({'response': llm_response.content})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
