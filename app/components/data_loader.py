import os


from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

from app.components.vector_store import save_vector_store
from app.components.pdf_loader import load_pdf_files, create_text_chunks

logger = get_logger(__name__)

def process_and_store_pdf_files():
    """
    Process and store PDF files in the vector store.
    """
    try:
        logger.info(f"Making Vector Store for PDF files with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}")

        documents = load_pdf_files()
        chunks = create_text_chunks(documents)
        save_vector_store(chunks)

        logger.info("Vector Store created successfully for PDF files.")

    except Exception as e:
        error_message = CustomException(f"Failed to Create Vector Store for PDF files: {str(e)}")
        logger.error(str(error_message))


if __name__ == "__main__":
    process_and_store_pdf_files()

