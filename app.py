import os
import openai
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
# Assuming LangChain and its dependencies are installed
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_very_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
openai.api_key = os.getenv('OPENAI_API_KEY')

db = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global db
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the uploaded PDF file
        extracted_texts = extract_text_from_pdf(file_path)
        embeddings = OpenAIEmbeddings()
        # Assuming `process_text_to_db` is a function that processes the text,
        # generates embeddings, and stores them in `db`
        db = Chroma.from_documents(extracted_texts, embeddings, persist_directory="db")

        session['uploaded_file_path'] = file_path

        return redirect(url_for('home', message="File uploaded successfully"))


@app.route('/query', methods=['POST'])
def handle_query():
    query_text = request.form['query']
    if db:
        # Configuration for Smaug via Hugging Face
        llm = OpenAI(temperature=0)
        # Retrieval from db based on similarity to find relevant text for the query
        retriever = db.as_retriever(search_kwargs={'k': 3})
        # Setup RetrievalQA with Smaug and the retriever
        rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                          return_source_documents=True, verbose=False,)

        # Execute the query and get the result
        response = rqa(query_text)['result']

        return response
    return jsonify(error="Database not initialized or query failed.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    return texts


if __name__ == '__main__':
    app.run(debug=True)
