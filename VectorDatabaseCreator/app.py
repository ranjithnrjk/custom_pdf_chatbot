import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma



async def extract_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

async def combine_texts_async(documents):
    tasks = [extract_text(pdf) for pdf in documents]
    texts = await asyncio.gather(*tasks)
    return ''.join(texts)

# Example usage
def get_text_chunks(text: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], # default separators, can add custom ones
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks =  text_splitter.create_documents([text])

    return chunks

def get_vectorstore(text_chunks):
    
    # embeddigns = OpenAIEmbeddings()
    # # create the open-source embedding function
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    db = Chroma.from_documents(text_chunks, embeddings, persist_directory="./db")
    


documents = os.listdir('./data')  # Replace with actual file paths
documents = [f'./data/{doc}' for doc in documents]

text = asyncio.run(combine_texts_async(documents))

chunks = get_text_chunks(text)  

get_vectorstore(chunks)