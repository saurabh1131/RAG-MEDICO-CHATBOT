from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get the embedding model based on the specified model name.

    Args:
        model_name (str): The name of the embedding model to use.

    Returns:
        HuggingFaceEmbeddings: An instance of the HuggingFaceEmbeddings class.
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully loaded embedding model: {model_name}")
        return embedding_model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise CustomException(f"Failed to load embedding model '{model_name}'", e) from e