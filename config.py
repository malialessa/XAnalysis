# config.py
import os

# Google Cloud / Vertex AI
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "br-ventasbrasil-cld-01")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") # ou 'southamerica-east1' se disponível para Gemini

# Google Sheets
GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL", "https://docs.google.com/spreadsheets/d/13hwbIhqHSqcF8oPmCs8OYo3KY732APIVmKRfeBf9BtM/edit?gid=1116222026#gid=1116222026")
GOOGLE_SHEET_TAB_NAME = os.getenv("GOOGLE_SHEET_TAB_NAME", "DataFunction")

# OCR Tesseract (opcional, se não estiver no PATH ou para depuração local)
# No Dockerfile, Tesseract já estará no PATH.
OCR_TESSERACT_PATH = os.getenv("OCR_TESSERACT_PATH", None) # Ex: r'C:\Program Files\Tesseract-OCR\tesseract.exe'