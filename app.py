from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate




# Initialize the SentenceTransformer model
embedding_function = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
chroma_db = "chroma_dbs"
llm = Ollama(model="qwen2:0.5b")

def read_and_store_in_chroma(pdf_path, chroma_db, embedding_function):
    # Initialize the PDF loader
    loader = PyMuPDFLoader(pdf_path)

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
    )

    # Load and split pages from the PDF
    pages = loader.load_and_split()
    chunks = text_splitter.split_documents(pages)

    # Store documents and embeddings in Chroma
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=chroma_db)
    db.persist()

def query_chroma(chroma_db, embedding_function, query, num_results=5):
    # Load the existing Chroma DB
    db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

    # Query the database
    docs = db.similarity_search(query, k=num_results)
    return docs

# Store the PDF documents in Chroma
#pdf_path = '1211.pdf'
#read_and_store_in_chroma(pdf_path, chroma_db, embedding_function)

# Query the stored documents in Chroma
query = "what is Admission Procedure"
docs = query_chroma(chroma_db, embedding_function, query)
#print(docs)
#print(docs[0].page_content)
combined_docs_content = "\n".join([doc.page_content for doc in docs])

print("doc_ouput: ", combined_docs_content)
# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a bot"), ("human", combined_docs_content)]
)


# Create the chain
chain = prompt_template | llm 


# Invoke the LLM with the combined document content
ai_response = chain.invoke({"query": query})

# Print the response from the LLM
print("Ai: ",ai_response)