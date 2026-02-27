import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from docx import Document
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Android app requests

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'txt', 'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# increase upload limit to 50MB (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Load API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Create a .env file with your API key.")

client = OpenAI(api_key=api_key)

# Store loaded documents
loaded_documents = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error reading document: {str(e)}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading document: {str(e)}"


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        return f"Error reading PDF: {e}"

from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Return JSON when upload exceeds configured limit"""
    return jsonify({'error': 'File too large. Maximum upload size is 50MB.'}), 413

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'Server is running'}), 200

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload a document (DOCX, TXT, or PDF)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use DOCX, TXT, or PDF'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type.  If a PDF is uploaded but a DOCX
        # with the same base name already exists in the upload folder, prefer
        # the DOCX version.
        if filename.lower().endswith('.docx'):
            doc_text = extract_text_from_docx(file_path)
        elif filename.lower().endswith('.pdf'):
            # check for existing docx equivalent
            base = filename[:-4]
            docx_equiv = os.path.join(app.config['UPLOAD_FOLDER'], base + '.docx')
            if os.path.exists(docx_equiv):
                doc_text = extract_text_from_docx(docx_equiv)
            else:
                doc_text = extract_text_from_pdf(file_path)
        else:
            doc_text = extract_text_from_txt(file_path)
        
        # Store document
        loaded_documents[filename] = doc_text
        
        return jsonify({
            'success': True,
            'message': f'Document "{filename}" uploaded successfully',
            'document_id': filename
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - sends question with document context to OpenAI"""
    try:
        data = request.json
        # debug log payload
        print(f"Chat request payload: {data}")
        user_message = data.get('message', '')
        document_id = data.get('document_id', None)
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get document content
        doc_text = ""
        if document_id and document_id in loaded_documents:
            doc_text = loaded_documents[document_id]
        elif loaded_documents:
            # Use first loaded document if no ID specified
            doc_text = list(loaded_documents.values())[0]
        else:
            return jsonify({'error': 'No document loaded. Please upload a document first.'}), 400
        
        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about the provided document. Answer based only on the document content."
                },
                {
                    "role": "user",
                    "content": f"Document content:\n{doc_text}\n\nUser question: {user_message}"
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        bot_reply = response.choices[0].message.content
        
        return jsonify({
            'success': True,
            'reply': bot_reply,
            'document_id': document_id
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all loaded documents"""
    documents = list(loaded_documents.keys())
    return jsonify({'documents': documents}), 200

@app.route('/api/clear', methods=['POST'])
def clear_documents():
    """Clear all loaded documents"""
    loaded_documents.clear()
    return jsonify({'message': 'All documents cleared'}), 200

if __name__ == '__main__':
    print("=" * 50)
    print("Starting OpenAI Bot Server")
    print("=" * 50)
    print("Server running on http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  POST   /api/upload   - Upload a document")
    print("  POST   /api/chat     - Send message and get reply")
    print("  GET    /api/documents - List loaded documents")
    print("  POST   /api/clear    - Clear all documents")
    print("  GET    /api/health   - Health check")
    print("=" * 50)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))