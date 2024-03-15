from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import CharacterTextSplitter
from google.cloud import storage
from langchain_google_vertexai import VertexAIEmbeddings
import variables as v
from langchain.document_loaders import GCSFileLoader
from PyPDF2 import PdfReader
import streamlit as st
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain.llms import VertexAI
import uuid
import re


def check_vectorstore(bucket_name):
  """
  Checks if a Google Cloud Storage bucket contains any blobs (files).

  :param bucket_name: The name of the Google Cloud Storage bucket to check.
  :return: True if the bucket contains at least one blob (file), False otherwise.
  """
  client = storage.Client()
  bucket = client.get_bucket(bucket_name)
  blobs = bucket.list_blobs()
  for blob in blobs:
      return True
  return False



class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

def get_pdf_docs(pdf_docs):
    """
    Reads and concatenates text content from a list of pdf files.

    :param pdf_docs: A list of paths to PDF documents.
    :return: A list of Document objects containing the text from each page and its metadata.
    """
    docs = []
    for pdf_path in pdf_docs:
        pdf_reader = PdfReader(pdf_path)
        page_number = 1
        for page in pdf_reader.pages:
            text = page.extract_text() if page.extract_text() else ""
            text = re.sub(r'(?:\r\n|\r|\n)', r'\\n', text)
            metadata = {'source': pdf_path.name, 'page': page_number}
            docs.append(Document(page_content=text, metadata=metadata))
            page_number += 1
    return docs


def get_docs(files):
    docs = []
    invalid = []
    for file in files:
        if ".txt" in file.name:
            text_doc = get_txt_text(file)
            
            text_splitter = CharacterTextSplitter(
                separator="",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            metadata = {'source':file.name,'format':'txt'}
            texts = text_splitter.create_documents([text_doc])
            for text in texts:
                text.metadata =metadata
            docs.extend(texts)
        elif ".pdf" in file.name:
            pdf_reader = PdfReader(file)
            page_number = 1
            for page in pdf_reader.pages:
                text = page.extract_text() if page.extract_text() else ""
                text = re.sub(r'(?:\r\n|\r|\n)', r' ', text)
                metadata = {'source': file.name, 'page': page_number,'format':'pdf'}
                docs.append(Document(page_content=text, metadata=metadata))
                page_number += 1
        else:
            invalid.append(file.name)
    if invalid:
        warning_message = "The following files do not have a valid format and will not be processed:\n"
        for file in invalid:
            warning_message += f"- {file}\n"
        st.warning(warning_message)
    return docs
    


def get_vectorstore_index(docs,embeddings):
    index= FAISS.from_documents(docs, embeddings)
    return index
 

def get_pdf_text(pdf_docs):
    """
    Reads and concatenates text content from a list of pdf files.

    :param pdf_docs: A list of paths to PDF documents.
    :return: A single string containing the concatenated text from all the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_text(txt_file):
    """
    Reads and concatenates text content from a list of text files.

    :param txt_files: A list of paths to text files.
    :return: A single string containing the concatenated text from all the text files.
    """
    file_content = txt_file.read()
    text = file_content.decode('utf-8')
    return text


def get_text_chunks(text):
    """
    Splits a long text into smaller chunks for processing.

    :param text: The input text to be split into smaller chunks.
    :return: A list of smaller text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Creates a vector store from text chunks using embeddings.

    :param text_chunks: A list of text chunks to be transformed into vectors.
    :return: A vector store containing vectors representing the input text chunks.
    """
    embeddings = VertexAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def store_to_df(store):
    """
    Converts a vector store into a DataFrame for easier data manipulation.

    :param store: The vector store to be converted into a DataFrame.
    :return: A DataFrame containing vector store data in a structured format.
    """
    v_dict = store.docstore._dict  
    data_rows = []
    for k, v in v_dict.items():  
        doc_name = v.metadata['source'].split('/')[-1]
        page_number = v.metadata.get('page', None)
        if page_number is not None:  
            page_number += 1
        content = v.page_content
        row = {"chunk_id": k, "document": doc_name, "content": content}
        if page_number:
            row["page"] = page_number
        data_rows.append(row)
    vector_df = pd.DataFrame(data_rows)
    column_order = ["chunk_id", "document", "page", "content"] if 'page' in vector_df.columns else ["chunk_id", "document", "content"]
    vector_df = vector_df[column_order]
    return vector_df


def docs_bucket(bucket_name):
    """
    Create a vector store from PDF and TXT documents stored in a Google Cloud Storage bucket.

    :param bucket_name: The name of the Google Cloud Storage bucket containing PDF and TXT documents.
    :return: A vector store containing vectors of text documents along with their format metadata.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs() 
    
    docs = []
    for blob in blobs:
        file_name = blob.name
        file_format = file_name.split('.')[-1].lower()  
        if file_format == 'pdf':
            loader_func = PyPDFLoader
        elif file_format == 'txt':
            loader_func = TextLoader
        else:
            continue  # 
        
        loader = GCSDirectoryLoader(project_name=bucket_name, bucket=bucket_name, loader_func=loader_func, file_path=file_name)
        pages = loader.load_and_split()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs.extend([{'content': doc, 'format': file_format} for doc in text_splitter.split_documents(pages)])
    return docs


def create_vectorstore(bucket_name, embeddings):
    """
    Create a vector store from documents stored in a Google Cloud Storage bucket.
    Supports both PDF and TXT documents.

    :param bucket_name: The name of the Google Cloud Storage bucket containing the documents.
    :param embeddings: Embeddings used for vectorization.
    :return: A vector store containing vectors of text documents.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()  

    docs = []
    for blob in blobs:
        file_name = blob.name
        file_format = file_name.split('.')[-1].lower()
        if file_format == 'pdf':
            loader_func = PyPDFLoader
        elif file_format == 'txt':
            loader_func = TextLoader
        else:
            continue 

        loader = GCSFileLoader(project_name=v.project_name, bucket=bucket_name, blob=file_name, loader_func=loader_func)
        document_content = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs.extend(text_splitter.split_documents(document_content))
        for doc in docs:
            doc.metadata['format']= file_format

    if docs:
        download_path = '/tmp' 
        bucket = client.get_bucket(v.bucket_vs)
        index = FAISS.from_documents(docs, embeddings)
        index.save_local(download_path)
        bucket.blob("index.pkl").upload_from_filename(f"{download_path}/index.pkl")
        bucket.blob("index.faiss").upload_from_filename(f"{download_path}/index.faiss")
        return index
    else:
        raise ValueError("No valid PDF or TXT files found in the bucket.")

def load_index_from_gcs(bucket_name,embeddings):
    """
    Load a vector store index from Google Cloud Storage and update it.

    :param bucket_name: The name of the Google Cloud Storage bucket containing the index.
    :param embeddings: Embeddings used for vectorization.
    :return: An updated vector store index.
    """
    download_path = '/tmp' 
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob_pkl = bucket.blob("index.pkl")
    blob_pkl.download_to_filename("/tmp/index.pkl")
    blob_faiss = bucket.blob("index.faiss")
    blob_faiss.download_to_filename("/tmp/index.faiss")
    index= FAISS.load_local(download_path, embeddings)
    uptadate_vectorestore_bucket(v.bucket_name,index,embeddings)
    index.save_local(download_path)
    blob_pkl.upload_from_filename("/tmp/index.pkl")
    blob_faiss.upload_from_filename("/tmp/index.faiss")
    return index


def delete_document(store, document):
    """
    Delete document chunks from a vector store based on the document name.

    :param store: The vector store from which document chunks should be deleted.
    :param document: The name of the document whose chunks need to be deleted.
    """
    vector_df = store_to_df(store)
    chunks_list = vector_df.loc[vector_df['document']==document]['chunk_id'].tolist()
    store.delete(chunks_list)


def refresh_model(new_store):
    """
    Refreshes a retrieval model with a new vector store.

    :param new_store: The new vector store to be used for retrieval.
    :return: A refreshed retrieval model with the specified vector store.
    """
    llm = VertexAI(
    model_name='text-bison@002',
    max_output_tokens=v.output_tokens,
    temperature=0.1,
    top_p=0.8,top_k=40,
    verbose=True,
    )
    retriever = new_store.as_retriever()
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return model


def add_to_vector_store(blob_doc, store, embeddings):
    """
    Adds a document from a Google Cloud Storage blob to an existing vector store.

    :param blob_doc: The Google Cloud Storage blob containing the document to be added.
    :param store: The existing vector store where the document will be added.
    :param embeddings: Embeddings used for vectorization.
    """
    file_format = (blob_doc.split('.')[-1].lower())
    if file_format == 'pdf':
        loader = GCSFileLoader(project_name=v.project_name, bucket=v.bucket_name, blob=blob_doc, loader_func=PyPDFLoader)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        for doc in docs:
            doc.metadata['format'] = 'pdf'
    elif file_format == 'txt':
        loader = GCSFileLoader(project_name=v.project_name, bucket=v.bucket_name, blob=blob_doc, loader_func=TextLoader)
        text_content = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(text_content)
        for doc in docs:
            doc.metadata['format'] = 'txt'
    extension = FAISS.from_documents(docs, embeddings)
    store.merge_from(extension)
    


def uptadate_vectorestore_bucket(bucket_name,store,embeddings):
    """
    Update a vector store based on the documents in a Google Cloud Storage bucket.

    :param bucket_name: The name of the Google Cloud Storage bucket containing documents.
    :param store: The vector store to be updated.
    :param embeddings: Embeddings used for vectorization.
    """
    vector_df = store_to_df(store)
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    blobs = list(bucket.list_blobs())
    documentos_eliminados = set(vector_df['document'].values) - set(blob.name for blob in blobs) 
    for blob in blobs:
      documento_nombre = blob.name
      if documento_nombre not in vector_df['document'].values:
          print("document added:",documento_nombre)
          add_to_vector_store(documento_nombre, store, embeddings)
          model = refresh_model(store)
    for documento_eliminado in documentos_eliminados:
          print("document deleted:",documento_eliminado)
          delete_document(store,documento_eliminado)
          model = refresh_model(store)
