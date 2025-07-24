# Dockerfile
# Use uma imagem base Python otimizada para Cloud Run
# Altere 'buster' para 'bookworm' (versão estável mais recente do Debian)
FROM python:3.10-slim-bookworm 

# Instale Tesseract OCR e suas dependências
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libpoppler-glib-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação para o diretório de trabalho
COPY . .

# Exponha a porta que a aplicação FastAPI vai rodar
ENV PORT 8080
EXPOSE 8080

# Comando para iniciar a aplicação com Uvicorn (servidor ASGI para FastAPI)
# O --host 0.0.0.0 é importante para que o contêiner escute em todas as interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]