from pathlib import Path
import streamlit as st
import os


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()
dir=os.path.dirname(__file__)

intro_markdown = read_markdown_file(str(dir)+"/CHANGELOG.md")

# Cambiar el tamaño del markdown utilizando CSS
markdown_style = f"<style> .stMarkdown {{ font-size: 16px; }} </style>"
st.markdown(markdown_style, unsafe_allow_html=True)

st.markdown(intro_markdown, unsafe_allow_html=True)