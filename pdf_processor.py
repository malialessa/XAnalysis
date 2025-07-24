# pdf_processor.py
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import os

# Configura o caminho do Tesseract se definido em config.py
# No ambiente Docker, Tesseract já estará no PATH
if os.getenv("OCR_TESSERACT_PATH"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("OCR_TESSERACT_PATH")

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """
    Extrai texto de um PDF. Tenta extrair texto diretamente.
    Se não for encontrado texto, tenta OCR para cada página.
    """
    document = fitz.open(pdf_file_path)
    full_text = []

    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text = page.get_text("text")

        if not text.strip(): # Se a página estiver vazia ou com espaços em branco, tenta OCR
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Tenta OCR em português
            ocr_text = pytesseract.image_to_string(img, lang='por')
            full_text.append(ocr_text)
        else:
            full_text.append(text)
    
    document.close()
    return "\n".join(full_text)

def save_uploaded_file_temp(uploaded_file_content: bytes, filename: str) -> str:
    """Salva o conteúdo do arquivo enviado em um arquivo temporário e retorna o caminho."""
    temp_dir = "/tmp" # Diretório temporário padrão no Cloud Run
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, filename)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file_content)
    return temp_file_path

def cleanup_temp_file(file_path: str):
    """Remove um arquivo temporário."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Arquivo temporário removido: {file_path}")

# Exemplo de uso (para teste local)
if __name__ == "__main__":
    # Crie um PDF de exemplo (para fins de teste)
    # Você precisaria de um PDF real para testar o OCR
    # Ex: crie um PDF com texto "normal" e outro com texto como imagem para testar OCR
    
    # Para testar OCR, você precisaria de um PDF escaneado ou uma imagem para simular
    # Exemplo com um PDF escaneado (substitua pelo seu arquivo real)
    # try:
    #     scanned_pdf_path = "Edital PEL04_2024.pdf [manifesto].pdf" # Assumindo que este é escaneado em parte
    #     if os.path.exists(scanned_pdf_path):
    #         print(f"Extraindo texto de {scanned_pdf_path}...")
    #         text = extract_text_from_pdf(scanned_pdf_path)
    #         print("--- EXTRAÇÃO DE TEXTO ---")
    #         print(text[:1000]) # Primeiros 1000 caracteres
    #         print("--- FIM DA EXTRAÇÃO ---")
    #     else:
    #         print(f"Arquivo {scanned_pdf_path} não encontrado para teste de OCR.")
    # except Exception as e:
    #     print(f"Erro ao testar OCR: {e}")
    #     print("Certifique-se de que o Tesseract está instalado e no PATH, ou configure OCR_TESSERACT_PATH no config.py.")
    pass