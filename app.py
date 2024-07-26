from langchain_community.chat_message_histories import SQLChatMessageHistory
from flask import Flask, request, render_template, jsonify
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama

chat_message_history = SQLChatMessageHistory(
    session_id="1", connection_string="sqlite:///sqlite.db"
)

#Define Model
llm = ChatOllama(model="qwen2:0.5b")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm


chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query_text", "")

    #chat_message_history.add_user_message(query_text)
    #chat_message_history.add_ai_message("Hi")
    config = {"configurable": {"session_id": "1"}}
    chain_with_history.invoke({"question": query_text}, config=config)




    messages = chat_message_history.messages
    print(messages, "\n\n")

    return jsonify({'results': messages}), 200




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
