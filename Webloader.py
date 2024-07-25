from langchain_community.document_loaders import WebBaseLoader

def get_response(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents    

print(get_response("https://www.pitambari.com/aboutus.php"))