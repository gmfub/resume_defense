import base64

import docx
import pdfplumber


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
    return text


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


# AI Vision for Images (The OCR Feature)
def extract_text_from_image(image_file, client):
    # Encode image to base64
    image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Send to Llama Vision (Groq)
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe the text from this resume image exactly. Do not summarize.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error reading image: {str(e)}"


# Master Router Function
def smart_parser(uploaded_file, client=None):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        if client:
            return extract_text_from_image(uploaded_file, client)
        else:
            return "Error: AI Client needed for images."
    return ""
