from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

# Load .env file
load_dotenv()

# Correct keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAPI_API_KEY1")   # Name fixed

# Export keys to system environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAPI_API_KEY1"] = OPENAI_API_KEY   # corrected name

# Load embeddings
embeddings = download_embeddings()

# Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# LLM
chatModel = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Default route
@app.route("/")
def index():
    return render_template("chat.html")

# Chat route
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])

    return response["answer"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7070, debug=True)
