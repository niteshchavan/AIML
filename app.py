from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Directory for Chroma persistence
db_directory = 'chromadb'

# Initialize the PDF loader
loader = PyPDFLoader('121.pdf')

# Load and split pages from the PDF
pages = loader.load_and_split()

# Initialize text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Define a function to split page content into lines
def split_lines(page_content):
    return page_content.split('\n')

# Split documents into chunks and then split each chunk into lines
documents = []
for page in pages:
    lines = split_lines(page.page_content)
    documents.extend(lines)

# Encode documents into embeddings
embeddings = model.encode(documents, convert_to_tensor=True)

# Store documents and embeddings in Chroma
chroma_db = Chroma.from_documents(documents, embeddings, persist_directory=db_directory)

# Print the content of the first page or chunk
if pages:
    print(pages[0].page_content)
else:
    print("No pages found or loaded from the PDF.")

# Ensure to handle errors and exceptions appropriately for production use.
