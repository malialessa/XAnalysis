# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import io

from pdf_processor import extract_text_from_pdf, save_uploaded_file_temp, cleanup_temp_file
from google_sheets_integrator import get_google_sheet_data
from ai_analyzer import initialize_vertex_ai, extract_requirements_with_gemini, cross_reference_assets
from config import GOOGLE_CLOUD_PROJECT_ID, GOOGLE_CLOUD_LOCATION, GOOGLE_SHEET_URL, GOOGLE_SHEET_TAB_NAME


app = FastAPI(
    title="Xertica - Análise Inteligente de Edital",
    description="API para análise de editais usando IA e integração de dados de ativos."
)

@app.on_event("startup")
async def startup_event():
    try:
        initialize_vertex_ai(GOOGLE_CLOUD_PROJECT_ID, GOOGLE_CLOUD_LOCATION)
        print("Vertex AI inicializado com sucesso.")
    except Exception as e:
        print(f"Erro ao inicializar Vertex AI: {e}")
        # Em um ambiente de produção, considere um 'sys.exit(1)' aqui
        # se a falha na inicialização da IA for um impedimento crítico.

@app.post("/analyze_edital/")
async def analyze_edital_endpoint(
    edital_file: UploadFile = File(..., description="Arquivo do edital (PDF ou DOCX).")
    # Removidos google_sheet_url e google_sheet_tab_name como parâmetros de Form
):
    """
    Endpoint para analisar um edital.
    Recebe um arquivo de edital (PDF/DOCX).
    A URL da planilha e o nome da aba são lidos das variáveis de ambiente do Cloud Run.
    Retorna uma análise estratégica e um mapa de atendimento.
    """
    
    # 1. Salvar o arquivo do edital temporariamente
    if edital_file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Apenas PDF e DOCX são aceitos.")

    temp_file_path = None
    try:
        file_content = await edital_file.read()
        temp_file_path = save_uploaded_file_temp(file_content, edital_file.filename)

        edital_text = ""
        if edital_file.content_type == "application/pdf":
            edital_text = extract_text_from_pdf(temp_file_path)
        elif edital_file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raise HTTPException(status_code=501, detail="Processamento de DOCX não implementado ainda. Use PDF.")


        if not edital_text.strip():
            raise HTTPException(status_code=400, detail="Não foi possível extrair texto do edital. O arquivo pode estar vazio ou ilegível.")

        # 2. Carregar dados da planilha de ativos das variáveis de ambiente
        try:
            # Usamos as variáveis de ambiente importadas de config.py
            ativos_df = get_google_sheet_data(GOOGLE_SHEET_URL, GOOGLE_SHEET_TAB_NAME)
            if ativos_df.empty:
                print("Aviso: Planilha de ativos vazia ou não pôde ser carregada. Apenas extração de requisitos será feita.")
        except Exception as e:
            print(f"Erro ao carregar planilha de ativos: {e}")
            ativos_df = pd.DataFrame() # Continue com um DataFrame vazio se houver erro

        # 3. Processamento Inteligente (Pipeline de IA)
        extracted_requirements = extract_requirements_with_gemini(edital_text)
        
        analysis_map_df = pd.DataFrame(columns=['Requisito', 'Tipo', 'Status', 'Evidência', 'Ação Necessária'])
        if not ativos_df.empty and extracted_requirements:
            analysis_map_df = cross_reference_assets(extracted_requirements, ativos_df)
        
        strategic_analysis = {
            "Objeto": extracted_requirements.get("Objeto", "N/A"),
            "Orgao": extracted_requirements.get("Orgao", "N/A"),
            "TipoJulgamento": extracted_requirements.get("TipoJulgamento", "N/A"),
            "ValorEstimado": extracted_requirements.get("ValorEstimado", "N/A"),
            "Datas": extracted_requirements.get("Datas", {})
        }

        analysis_map_json = analysis_map_df.to_dict(orient="records")

        resumo_requisitos = {}
        for k, v in extracted_requirements.get("RequisitosHabilitacao", {}).items():
            if v:
                resumo_requisitos[f"Habilitação {k}"] = v
        for item in extracted_requirements.get("RequisitosObjetoQualificacaoTecnicaEspecifica", []):
            desc = item.get("Descricao", "")
            details = ", ".join(item.get("Detalhes", []))
            quant = item.get("QuantitativoMinimo", "")
            cert = item.get("CertificacaoExigida", "")
            resumo_requisitos[f"Objeto/Técnico: {desc}"] = f"{details} (Quant.: {quant}, Cert.: {cert})"


        return JSONResponse(content={
            "analysis_strategic": {
                "NomeOrgao": strategic_analysis.get("Orgao"),
                "Objeto": strategic_analysis.get("Objeto"),
                "CriterioJulgamento": strategic_analysis.get("TipoJulgamento"),
                "ValorEstimado": strategic_analysis.get("ValorEstimado"),
                "Datas": strategic_analysis.get("Datas"),
                "ResumoRequisitosExtraidos": resumo_requisitos
            },
            "mapa_atendimento": analysis_map_json
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erro inesperado no endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)

# Para rodar localmente com Uvicorn (para testes)
# if __name__ == "__main__":
#     import uvicorn
#     # Para testar localmente, 'gcloud auth application-default login' deve ser executado
#     # e sua conta de usuário deve ter as permissões necessárias (Vertex AI User, Sheets Reader)
#     os.environ["GOOGLE_CLOUD_PROJECT_ID"] = "seu-projeto-id"
#     os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1" # Ou sua região para Gemini
#     os.environ["GOOGLE_SHEET_URL"] = "https://docs.google.com/spreadsheets/d/13hwbIhqHSqcF8oPmCs8OYo3KY732APIVmKRfeBf9BtM/edit?gid=1116222026#gid=1116222026"
#     os.environ["GOOGLE_SHEET_TAB_NAME"] = "DataFunction"
#     uvicorn.run(app, host="0.0.0.0", port=8000)