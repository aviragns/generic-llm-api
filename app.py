import os
from pathlib import Path
from dotenv import load_dotenv
import google.auth
import logging
from flask import Flask, jsonify, make_response, request
# import the right LLM 
from langchain.llms import VertexAI
# import the right tools for data ingestion and cleansing
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import the right tools for embedding and vectordb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# import the right chain for VA
from langchain.chains import RetrievalQA
# Disable chroma telemetry

logging.basicConfig(encoding='utf-8', level=logging.INFO)

load_dotenv()

# Pre-check on environments
credentials, project_id = google.auth.default()
if(project_id != os.environ.get("GCP_PROJECT_ID")):
    logging.error("Project ID is NOT correct")
else:
    logging.info("ADC is setup properly.")
if(not Path(os.environ.get("DOCS_PATH")).is_dir()):
    logging.error("DOCS_PATH not found")

# get_llm
def get_llm() -> any:
    return(VertexAI(model_name='text-bison'))

# get_splitter
def get_splitter() -> any:
    return RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=20
    )

# get_vectordb
def get_vectordb() -> any:
    # Run through the documents in DOCS_PATH
    text_splitter = get_splitter()
    texts = list()
    p = Path(os.environ.get("DOCS_PATH"))
    for f in p.glob('*.pdf'):
        target = str(f.resolve())
        loader = PyPDFLoader(target)
        logging.info(f"Processing {target}")
        pages = loader.load()
        logging.info(f"there are {len(pages)} pages")
        texts += text_splitter.split_documents(pages)
        logging.info(f"being splitted into {len(texts)} documents")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectordb = Chroma.from_documents(texts, hf)
    return vectordb

llm = get_llm()
vectordb = get_vectordb()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(), return_source_documents=True)

# building the flask app
app = Flask(__name__)
@app.route("/", methods = ['POST'])
def answer():
    if not request.is_json:
        return make_response(
            jsonify(
                {"success": False,
                 "error": "Unexpected error, request is not in JSON format"}),
            400)
    try:
        data = request.json
        message = data["message"]
        logging.info(f"message received: {message}")
        resp = qa(message)
        logging.info(resp)
        source_documents = [d.__dict__ for d in resp['source_documents']]
        return jsonify({
            'query': resp['query'],
            'result': resp['result'],
            'source_documents': source_documents
        })
    except Exception as err:
        logging.error(f"Unexpected {err=}, {type(err)=}")
        logging.error(f"request received: {data}")
        return make_response(
            jsonify(
                {"success": False, 
                 "error": "Unexpected error: failed to send the message"}),
            400)

if __name__ == "__main__":
    app.run()