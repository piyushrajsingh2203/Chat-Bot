from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import pytesseract
import os
import tempfile
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables.
load_dotenv()
app = Flask(__name__)

# MongoDB connection.
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client['chatbotDB']
collection = db['documents']

# OpenAI Embeddings.
embedding = OpenAIEmbeddings()

# Text splitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# ChromaDB setup.
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="chroma_storage"
)

# extract text from files.
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

def extract_text_from_image(file_path):
    return pytesseract.image_to_string(file_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files['file']
        file_ext = file.filename.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

            if file_ext == 'pdf':
                text = extract_text_from_pdf(temp_file_path)
            elif file_ext in ['jpg', 'jpeg', 'png']:
                text = extract_text_from_image(temp_file_path)
            else:
                text = file.read().decode("utf-8")

        os.unlink(temp_file_path)

        # Split text into chunks.
        text_chunks = text_splitter.split_text(text)

        # Store document metadata and content in MongoDB.
        doc_id = collection.insert_one({
            "text": text,
            "file_name": file.filename,
            "upload_date": datetime.utcnow()
        }).inserted_id

        # Store chunks in Chroma.
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(text_chunks))]
        vector_store.add_texts(
            texts=text_chunks,
            ids=chunk_ids,
            metadatas=[{"original_doc_id": str(doc_id)} for _ in text_chunks]
        )

        return jsonify({
            "status": "Text extracted and stored",
            "document_id": str(doc_id),
            "chunks": len(text_chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

@app.route("/query", methods=["POST"])
def query():
    try:
        user_query = request.json.get("query")
        
        # Retrieve chunks from ChromaDB.
        results = vector_store.similarity_search(
            user_query,
            k=3  
        )
        
        
        prompt_template = """
        Answer the question based on the context provided. If you cannot find the answer in the context, say "I cannot find the answer in the provided context."
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        llm = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            max_tokens=500,
            temperature=0.3
        )
        
    
        chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            prompt=PROMPT
        )
        
       
        answer = chain.run(
            input_documents=results,
            question=user_query
        )
        
        return jsonify({
            "answer": answer,
            "chunks_used": len(results)
        })

        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)