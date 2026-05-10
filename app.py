import os
import fitz
import numpy as np
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# Simple in-memory store instead of ChromaDB
chunks_store = []

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

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global chunks_store
    if 'file' not in request.files:
        return jsonify({'success': False})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    text = extract_text_from_pdf(filepath)
    chunks = chunk_text(text)
    chunks_store = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        chunks_store.append({'text': chunk, 'embedding': embedding})
    return jsonify({'success': True, 'chunks': len(chunks)})

@app.route('/ask', methods=['POST'])
def ask():
    global chunks_store
    if not chunks_store:
        return jsonify({'answer': 'Please upload a PDF first.'})
    data = request.get_json()
    question = data.get('question', '')
    question_embedding = get_embedding(question)
    similarities = []
    for chunk in chunks_store:
        sim = cosine_similarity(question_embedding, chunk['embedding'])
        similarities.append((sim, chunk['text']))
    similarities.sort(reverse=True)
    top_chunks = [text for _, text in similarities[:3]]
    context = " ".join(top_chunks)
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