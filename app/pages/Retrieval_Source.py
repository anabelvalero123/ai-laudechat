from pathlib import Path
import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import fitz
from streamlit_extras.stylable_container import stylable_container
from google.cloud import storage
import variables as v

def pdf_page_to_image(pdf_file, page_number):
    """
    Converts a specific page of a PDF file into an image.

    :param pdf_file: The PDF file as a file-like object.
    :param page_number: The number of the page to convert.
    :return: The image resulting from the conversion of the specified page.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        images = convert_from_path(tmp_file.name, first_page=page_number, last_page=page_number)
    return images[0]  


def get_score_by_page(documents_and_scores, target_page):
    """
    Retrieves the score of a document based on its page number.

    :param documents_and_scores: A list of tuples (document, score).
    :param target_page: The target page number.
    :return: The score of the document matching the page number, or None if not found.
    """
    for document, score in documents_and_scores:
        if document.metadata['page'] == target_page:
            return score
    return None


def download_file(source):
    """
    Downloads a file from a specified source, which can be a Google Cloud Storage link or a local file.

    :param source: The file's source, which can be a Google Cloud Storage URL or a local file name.
    :return: The path to the downloaded file.
    """
    if "gs" in source:
        nombre_archivo = os.path.basename(source)
        file_path = "tmp_docs/" + nombre_archivo 
        if not os.path.exists(file_path):
            client = storage.Client()
            bucket = client.get_bucket(v.bucket_name)
            blob = bucket.blob(nombre_archivo)
            blob.download_to_filename(file_path)
    else:
        file_path = "tmp_docs/" + source
        if not os.path.exists(file_path):
            doc = [file for file in st.session_state.documents if file.name == source]
            doc[0].seek(0) 
            with open(file_path, "wb") as f:
                f.write(doc[0].read()) 
    return file_path


def save_image_from_page(file_path, num_page):
    """
    Saves an image of a specific page from a PDF document.

    :param file_path: The path to the PDF file.
    :param num_page: The page number to convert into an image.
    :return: The name of the saved image.
    """
    documento = fitz.open(file_path)
    pagina = documento.load_page(num_page)
    matriz = fitz.Matrix(4.0, 4.0)
    pix = pagina.get_pixmap(matrix=matriz)
    img_name = "tmp_docs/img_" + str(num_page) + ".png"
    pix.save(img_name)
    documento.close()
    return img_name


if 'response' not in st.session_state:
    st.session_state.response =None
elif st.session_state.response is not None:
    with open("archivo.txt", "w", encoding="utf-8") as archivo:
        archivo.write(str(st.session_state.response))
    Docs = st.session_state.response
    st.header("Retrival Source")
    Query = f"**User Question:**\n {st.session_state.last_user_msg}"
    Answer = f"**Bot Answer:**\n {st.session_state.last_bot_msg}"
    Num_sources =  f"**Number of sources :** {len(Docs)}"
    #docs_and_scores = st.session_state.vectorstore.similarity_search_with_score(st.session_state.last_user_msg)
    st.markdown(f"{Query}\n\n{Answer}\n\n{Num_sources}")   
    for Document in Docs:
        st.divider() 
        if Document.metadata['format']=='pdf':
            #score = get_score_by_page(docs_and_scores, Document.metadata['page'])
            #score_text = f"**Score:**\n {format(score, '.4f')}"
            source_text = f"**Source:**\n {Document.metadata['source']}"
            #page_text = f"**Page:**\n {Document.metadata['page']}"
            page_content_text = f"**Page content:**\n {Document.page_content}"
            with stylable_container(
                key="container_with_border",
                    css_styles="""
                        {
                            border: 1px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px)
                        }
                        """,
                ):
                    st.markdown(f"{source_text}")
                    

        elif Document.metadata['format']=='txt':
            source_text = f"**Source:**\n {Document.metadata['source']}"
            page_content_text = f"**Page content:**\n {Document.page_content}"
            with stylable_container(
                key="container_with_border",
                    css_styles="""
                        {
                            border: 1px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px)
                        }
                        """,
                ):
                    st.markdown(f"{source_text}")

        with st.expander("OCR"):
            st.markdown(Document.page_content)

        if Document.metadata['format']=='pdf':
            with st.expander("Document Page"):
                if not os.path.exists("tmp_docs"):
                    os.makedirs("tmp_docs", exist_ok=True)
                if "gs" in Document.metadata['source']:
                    num_page = Document.metadata['page']
                else:
                    num_page = Document.metadata['page']-1 
                file_path = download_file(Document.metadata['source'])
                img_name = save_image_from_page(file_path, num_page)
                st.image(img_name)



