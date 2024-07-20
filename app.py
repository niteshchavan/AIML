from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer, util
from langchain_chroma import Chroma



# Load a lightweight model
model = SentenceTransformer('all-mpnet-base-v2')

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    return ' '.join([page.page_content for page in data])

# Path to your PDF file
pdf_path = '121.pdf'

doc = extract_text_from_pdf(pdf_path)

docs = doc.split('\n')


print(docs)