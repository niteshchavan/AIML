from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma



loader = PyPDFLoader('121.pdf')

pages = loader.load_and_split()


print(pages[0])
