from openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv
import fitz
import os

def extract_clean_text_fitz(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        cleaned_text = "\n".join([line.rstrip() for line in page_text.split("\n")])  # Remove trailing spaces
        text += cleaned_text + "\n"
    return text.strip()



pdf_path = "10_CE106_CE127_ProjectReport.docx.pdf"  # Change this to your file path
text = extract_clean_text_fitz(pdf_path)
print(text)


load_dotenv()

api_key = os.getenv("BYTEDANCE_APIKEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  model="bytedance-research/ui-tars-72b:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": text + "give answers from this text only of below questions"
        },
        {
            
          "type": "text",
          "text": "Explain SVM"
        
        },
       
        
      ]
    }
  ]
)
print(completion.choices[0].message.content)