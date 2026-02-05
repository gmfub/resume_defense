import base64
import json
import logging
import os
from io import BytesIO
from typing import Dict, Optional, Union

import docx
import pdfplumber
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

# --- UTILS ---


def extract_text_from_pdf(pdf_file) -> str:
    """Extracts raw text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extract = page.extract_text()
                if extract:
                    text += extract + "\n"
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return f"Error reading PDF: {str(e)}"
    return text


def extract_text_from_docx(docx_file) -> str:
    """Extracts raw text from a DOCX file."""
    try:
        doc = docx.Document(docx_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return f"Error reading DOCX: {str(e)}"


def encode_image_base64(image_file) -> str:
    """Encodes a file object to base64 string."""
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode("utf-8")


# --- OPENAI VISION HANDLER ---


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_from_image_openai(
    image_file, client: OpenAI, mode: str = "text"
) -> Union[str, Dict]:
    """
    Extracts content from an image using OpenAI Vision (GPT-4o).

    Args:
        image_file: Streamlit file object.
        client: Initialized OpenAI client.
        mode: 'text' or 'jd_structured'.
    """
    if not client:
        return "Error: OpenAI Client not initialized."

    try:
        base64_image = encode_image_base64(image_file)

        # Determine strictness/prompt
        if mode == "jd_structured":
            system_prompt = "You are an expert HR Parser."
            user_prompt = """Extract data from this Job Description screenshot.
            Return a valid JSON object with the following structure:
            {
                "raw_text": "The full text...",
                "job_title": "Title",
                "company_name": "Company",
                "required_skills": ["Skill1", "Skill2"],
                "preferred_skills": ["Skill3"],
                "years_experience": "Experience req",
                "education": "Education req"
            }
            """
            response_format = {"type": "json_object"}
        else:
            system_prompt = "You are a helpful OCR assistant."
            user_prompt = "Transcribe the text from this document image exactly. Do not summarize."
            response_format = None

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq Llama 4 Vision
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            response_format=response_format,
            temperature=0.1,  # Low temp for deterministic output
        )

        content = response.choices[0].message.content

        if mode == "jd_structured":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "raw_text": content,
                    "job_title": "Error parsing JSON",
                    "required_skills": [],
                }

        return content

    except Exception as e:
        logger.error(f"OpenAI extraction error: {e}")
        if mode == "jd_structured":
            return {
                "raw_text": f"Error with OpenAI: {str(e)}",
                "job_title": "Error",
                "required_skills": [],
            }
        return f"Error reading image: {str(e)}"


# --- MASTER ROUTER ---


def smart_parser(
    uploaded_file, client: Optional[OpenAI] = None, is_jd=False
) -> Union[str, Dict]:
    """
    Routes the file to the appropriate extractor.
    NOTE: 'client' argument is now REQUIRED for image processing (OpenAI).
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        # If client is missing, we can try to initialize a temp one or error out
        if not client:
            # Fallback: try init from env
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except:
                return "Error: API Client missing for image processing."

        mode = "jd_structured" if is_jd else "text"
        return extract_from_image_openai(uploaded_file, client, mode=mode)

    return ""
