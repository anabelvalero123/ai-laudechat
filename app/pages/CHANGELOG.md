# Changelog

### [Unreleased]

* Multimodal Retrieval Augmented Generation (RAG).
* Use of Gemini.
* Use of document metadata.

### [0.3.0] - 2024-02-29

#### Added

* Similarity search with score to obtain retrieval sources.
* Streamlit page for retrieval source visualization, showcasing the documents names and the text segments used (OCR). For PDF documents, it also presents the specific page number and a visual representation of that page.
* Button to initiate a new session. Clicking this button will erase the current chat conversation, allowing for a fresh start.
 
#### Fixed
* Integration of PDF and TXT documents for simultaneous use of both sources.

### [0.2.0] - 2024-01-31

#### Added

* Content extraction from '.txt' documents.
* Warning message to alert users if the document format is not accepted (only PDF and txt are accepted).
* Warning message when the user asks a question after loading documents without clicking 'Process'.
* Chatbot configuration for setting the maximum output length in tokens (slider).
* Streamlit page for visualizing application updates and changes.

#### Changed

* Chat input placement change to keep the latest conversation messages in view without the need for scrolling

### [0.1.0] - 2024-01-20

#### Added

* Initial version of the Laude Chat Streamlit application (`LaudeChat.py`) with the following features:
  - Streamlit integration for user interface creation.
  - User input handling and response generation with VertexAI (text-bison@002).
  - Chat history display with user and bot messages.
* Integration with Google Cloud services.
* Utility functions for vector store management (`vectorstore_utils.py`) with the following features:
  - PDF content extraction and text chunking using Langchain.
  - Vector store creation (FAISS) from text chunks using embeddings (textembedding-gecko@003).
  - Updating a vector store based on documents in a Google Cloud Storage bucket.
* HTML templates for formatting chat messages (`htmlTemplates.py`).
* CSS styling for chat message display.
* Comprehensive README and CHANGELOG documentation for clarity and guidance.



