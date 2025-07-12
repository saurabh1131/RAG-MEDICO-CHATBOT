import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    """
    Load all PDF files from a specified directory and split them into chunks.

    Args:
        directory (str): The path to the directory containing PDF files.

    Returns:
        list: A list of document chunks.
    """
    try:
        directory = DATA_PATH

        if not os.path.exists(directory):
            raise CustomException(f"Directory {directory} does not exist.")
        logger.info(f"Loading PDF files from directory: {directory}")
        # Load PDF files from the directory
        loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            logger.warning("No PDF files found in the directory.")
            return []
        else:
            logger.info(f"Successfully loaded %d PDF files: {len(documents)}")

        return documents

    except Exception as e:
        logger.error(f"Error loading PDF files: {e}")
        return []


def create_text_chunks(documents):
    """
    Split documents into chunks using a text splitter.

    Args:
        documents (list): A list of documents to be chunked.

    Returns:
        list: A list of document chunks.
    """
    try:
        if not documents:
            logger.warning("No documents provided for chunking.")
            return []
        logger.info(f"Starting text chunking process. Lenght of documents: {len(documents)}")

        # Initialize the text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # length_function=len,
            # is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully chunked documents into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error during text chunking: {e}")
        raise CustomException(f"Failed to chunk documents: {e}") from e