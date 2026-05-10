import os
import fitz
import chromadb
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

chroma_client = chromadb.Client()
collection = None

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global collection
    if 'file' not in request.files:
        return jsonify({'success': False})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        chroma_client.delete_collection("docs")
    except:
        pass
    collection = chroma_client.get_or_create_collection(name="docs")
    text = extract_text_from_pdf(filepath)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"chunk_{i}"])
    return jsonify({'success': True, 'chunks': len(chunks)})

@app.route('/ask', methods=['POST'])
def ask():
    global collection
    if collection is None:
        return jsonify({'answer': 'Please upload a PDF first.'})
    data = request.get_json()
    question = data.get('question', '')
    results = collection.query(query_texts=[question], n_results=3)
    context = " ".join(results["documents"][0])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer questions using only the context provided. Be concise and helpful."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return jsonify({'answer': response.choices[0].message.content})

if __name__ == '__main__':
    app.run(debug=True)