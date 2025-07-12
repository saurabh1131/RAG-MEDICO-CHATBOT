from langchain.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate

from app.components.vector_store import load_vector_store

from app.components.llm import load_llm

from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)

CUSTOM_PROMPT = \
"""
Answer the following Medical Question based on the provided context in 2-3 lines max. If the answer is not found in the context, say "I don't know".

Context: 
{context}

Question: 
{question}

Answer:
"""

def set_custom_prompt():
    """
    Set the custom prompt for the RetrievalQA chain.
    """
    try:
        logger.info("Setting custom prompt for RetrievalQA chain.")
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT,
            input_variables=["context", "question"]
        )
        return prompt
    except Exception as e:
        logger.error(f"Error setting custom prompt: {e}")
        raise CustomException("Failed to set custom prompt", e) from e


def create_retrieval_qa_chain():
    """
    Create a RetrievalQA chain with the specified vector store and LLM.

    Args:
        vector_store_path (str): Path to the vector store.
        llm: The language model to use. If None, it will load the default LLM.

    Returns:
        RetrievalQA: An instance of the RetrievalQA chain.
    """
    try:
        logger.info("Loading vector store.")
        vector_store = load_vector_store()

        if vector_store is None:
            raise CustomException("Vector store not found. Please ensure it is initialized.")

        llm = load_llm()
        if llm is None:
            raise CustomException("LLM not loaded. Please ensure it is initialized.")

        logger.info("Creating RetrievalQA chain.")
        prompt = set_custom_prompt()
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        return retrieval_qa_chain
    except Exception as e:
        logger.error(f"Error creating RetrievalQA chain: {e}")
        raise CustomException("Failed to create RetrievalQA chain", e) from e