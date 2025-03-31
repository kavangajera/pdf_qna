from transformers import pipeline
import fitz  # PyMuPDF for reading PDFs

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def main():
    pdf_path = input("Enter PDF file path: ")
    context = extract_text_from_pdf(pdf_path)

    if not context.strip():
        print("No text found in PDF. Exiting...")
        return

    print("PDF loaded successfully. You can now ask questions.")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting... Goodbye!")
            break

        try:
            result = qa_pipeline(question=question, context=context)
            print(f"Answer: {result['answer']}")
        except Exception as e:
            print(f"Error in generating answer: {e}")

if __name__ == "__main__":
    main()
