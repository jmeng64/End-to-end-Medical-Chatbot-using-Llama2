from src.helper import load_pdf, text_split, download_hugging_face_embeddings
#from langchain.vectorstores import Pinecone
from langchain_pinecone import Pinecone  
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



#Initializing the Pinecone
# pinecone.init(api_key=PINECONE_API_KEY,
#             environment=PINECONE_API_ENV)

pc=pinecone.Pinecone(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV,
              region="us-east-1")


index_name="medical-chatbot"
index = pc.Index(index_name)

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
