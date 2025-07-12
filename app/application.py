import os

from flask import Flask, render_template, request, redirect, url_for, session

from app.components.retriever import create_retrieval_qa_chain

from app.common.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup

def nl2br(text):
    """
    Convert newlines in text to HTML <br> tags.
    """
    return Markup(text.replace('\n', '<br>'))


app.jinja_env.filters['nl2br'] = nl2br


logger = get_logger(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Render the main page and handle form submissions.
    """
    if "message" not in session:
        session["messages"] = []#["Welcome to the Medical Question Answering System! Please enter your question below."]

    if request.method == 'POST':
        user_input = request.form.get('prompt', '').strip()

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                qa_chain = create_retrieval_qa_chain()
                response = qa_chain.invoke({"query": user_input})
                result = response.get('result', "No answer found.")
                logger.info(f"User input: {user_input}")
                logger.info(f"Result: {result}")

                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages

            except Exception as e:
                logger.error(f"Error processing user input: {e}")
                result = "An error occurred while processing your question. Please try again later."
                return render_template('index.html', messages=session["messages"], result=result, error=e)

        return render_template('index.html', messages=session["messages"], result=result)

    return render_template('index.html', messages=session.get("messages", []))


@app.route('/clear')
def clear():
    """
    Clear the session messages.
    """
    session.pop("messages", None)
    session["messages"] = []  # ["Welcome to the Medical Question Answering System! Please enter your question below."]
    return redirect(url_for("index"))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5005, use_reloader=False)