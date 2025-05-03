from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import fitz  # PyMuPDF
from dotenv import load_dotenv
import os

# Initialize Flask app
app = Flask(__name__)

CORS(app)

# Load environment variables
load_dotenv(dotenv_path=".env")

# OpenAI API setup
api_key = os.getenv("BYTEDANCE_APIKEY")
if not api_key:
    raise ValueError("API key not found. Please check your .env file.")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Function to extract and clean text from a PDF
def extract_clean_text_fitz(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        cleaned_text = "\n".join([line.rstrip() for line in page_text.split("\n")])  # Remove trailing spaces
        text += cleaned_text + "\n"
    return text.strip()

# Flask route to handle the API request
@app.route('/api/find_answer', methods=['POST'])
def find_answer():
    try:
        # Get the PDF path and question from the request
        data = request.json
        pdf_path = data.get("pdf_path")
        question = data.get("question")

        if not pdf_path or not question:
            return jsonify({"error": "pdf_path and question are required"}), 400

        # Extract text from the PDF
        text = extract_clean_text_fitz(pdf_path)

        # Generate the answer using OpenAI API
        completion = client.chat.completions.create(
            model="bytedance-research/ui-tars-72b:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text + "give answers from this text only of below questions. If not found, say 'Not found in given PDF'."
                        },
                        {
                            "type": "text",
                            "text": question
                        },
                    ]
                }
            ]
        )

        # Return the answer in JSON format
        answer = completion.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)