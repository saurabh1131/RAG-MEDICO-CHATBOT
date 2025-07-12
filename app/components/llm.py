from langchain_huggingface import HuggingFaceEndpoint

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN, GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    """
    Load the LLM from Hugging Face Hub.

    Args:
        huggingface_repo_id (str): The repository ID of the model on Hugging Face.
        hf_token (str): The Hugging Face token for authentication.

    Returns:
        LLM: An instance of the loaded LLM.
    """
    try:
        # logger.info(f"Loading LLM from Hugging Face Hub: {huggingface_repo_id}")
        # llm = HuggingFaceEndpoint(
        #     repo_id=huggingface_repo_id,
        #     temperature=0.1,
        #     max_new_tokens=256,
        #     return_full_text = False,
        #     huggingfacehub_api_token=hf_token,
        #     task="conversational"
        # )
        logger.info(f"Loading LLM from Gemini API")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            max_output_tokens=256,
            temperature=0.1,
            top_p=0.95,
            api_key=GEMINI_API_KEY
        )
        logger.info("Successfully loaded LLM.")
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise CustomException("Failed to load LLM", e) from e