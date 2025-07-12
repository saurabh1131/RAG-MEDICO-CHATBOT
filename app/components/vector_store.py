import os.path

from langchain_community.vectorstores import FAISS

from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    """
    Load the FAISS vector store from the specified path.

    Returns:
        FAISS: An instance of the FAISS vector store.
    """
    try:
        logger.info(f"Loading FAISS vector store from path: {DB_FAISS_PATH}")
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info("FAISS vector store path exists, proceeding to load.")
            faiss_vector_store = FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True)
            logger.info("Successfully loaded FAISS vector store.")
            return faiss_vector_store
        else:
            logger.warning(f"FAISS vector store path does not exist: {DB_FAISS_PATH}")
    except Exception as e:
        logger.error(f"Error loading FAISS vector store: {e}")
        raise CustomException("Failed to load FAISS vector store", e) from e


def save_vector_store(text_chunks):
    """
    Save the FAISS vector store to the specified path.

    Args:
        faiss_vector_store (FAISS): The FAISS vector store to save.
    """
    try:
        if not text_chunks:
            logger.warning("No text chunks provided to save in FAISS vector store.")
            raise CustomException("No text chunks provided for saving.")

        logger.info(f"Creating FAISS vector store with {len(text_chunks)} text chunks.")

        db = FAISS.from_documents(
            text_chunks,
            get_embedding_model()
        )

        db.save_local(DB_FAISS_PATH)

        logger.info("Successfully saved FAISS vector store.")

        return db

    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        raise CustomException("Failed to create FAISS vector store", e) from e
