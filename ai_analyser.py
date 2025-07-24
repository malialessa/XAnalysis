# ai_analyzer.py
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
from vertexai.language_models import TextEmbeddingModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re

# Função para inicializar o Vertex AI
def initialize_vertex_ai(project_id: str, location: str):
    vertexai.init(project=project_id, location=location)

# Função para extrair requisitos usando Gemini
def extract_requirements_with_gemini(edital_text: str) -> dict:
    """
    Extrai informações estruturadas de requisitos do edital usando o modelo Gemini.
    """
    model = GenerativeModel("gemini-2.5-flash") # Ou "gemini-1.0-pro"

    # Prompt mais robusto e com exemplos para guiar a extração
    prompt = f"""
    Dado o seguinte texto de edital, extraia as seguintes informações no formato JSON.
    Se uma informação não for encontrada, use "N/A".
    Para datas e valores, extraia-os no formato mais padronizado possível.
    Identifique claramente os requisitos de habilitação (jurídica, técnica, fiscal, econômico-financeira) e os requisitos do objeto/qualificação técnica específicos do serviço/produto.
    
    Estrutura JSON esperada:
    {{
        "Objeto": "...",
        "Orgao": "...",
        "TipoJulgamento": "...",
        "ValorEstimado": "...",
        "Datas": {{
            "AcolhimentoPropostasInicio": "DD/MM/AAAA",
            "AcolhimentoPropostasFim": "DD/MM/AAAA HH:MM",
            "AberturaPropostas": "DD/MM/AAAA HH:MM",
            "DisputaPrecos": "DD/MM/AAAA HH:MM"
        }},
        "RequisitosHabilitacao": {{
            "Juridica": ["Prova de registro comercial", "Ato constitutivo/estatuto social", ...],
            "Fiscal": ["Prova de inscrição CNPJ", "Regularidade Fazenda Federal/Estadual/Municipal", "Regularidade Seguridade Social/FGTS", ...],
            "EconomicoFinanceira": ["Certidão Negativa de Falência", "Balanço Patrimonial com índices (LG, SG, LC)", "Patrimônio Líquido mínimo (se aplicável)", ...],
            "TecnicaGeral": ["Atestado(s) de capacidade técnica", "Certificações", "Comprovação de vínculo do profissional", ...]
        }},
        "RequisitosObjetoQualificacaoTecnicaEspecifica": [
            {{
                "Tipo": "Serviço/Produto",
                "Descricao": "...",
                "Detalhes": ["...", "..."],
                "QuantitativoMinimo": "...",
                "CertificacaoExigida": "...",
                "PrazosEspecificos": "...",
                "MencaoIA": "Sim/Não"
            }}
            // Pode haver múltiplos objetos/itens
        ],
        "CondicoesEspecializadas": {{
            "POC": "Sim/Não",
            "Garantias": ["...", "..."],
            "CriteriosEliminatorios": ["...", "..."],
            "SubcontratacaoPermitida": "Sim/Não"
        }},
        "InformacoesGerais": {{
            "ValidadeProposta": "...",
            "LocalEntregaDocs": "...",
            "ContatoDuvidas": "..."
        }}
    }}

    Texto do Edital:
    {edital_text}
    """

    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    
    # O Gemini pode envolver o JSON em '```json\n...\n```'. Limpar isso.
    try:
        json_string = response.text.strip()
        if json_string.startswith("```json") and json_string.endswith("```"):
            json_string = json_string[len("```json"): -len("```")].strip()
        
        # Correção de possíveis erros de formatação JSON comuns do LLM
        # Remover vírgulas penduradas antes de chaves de fechamento ou colchetes
        json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
        # Substituir aspas simples por duplas (se o modelo ocasionalmente usar aspas simples)
        json_string = json_string.replace("'", '"')
        # Tentar corrigir chaves de abertura/fechamento que podem estar incorretas
        json_string = re.sub(r'([a-zA-Z0-9_]+)\s*:\s*([^,{}\[\]"]+)', r'"\1": "\2"', json_string)


        extracted_data = json.loads(json_string)
        return extracted_data
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON do Gemini: {e}")
        print(f"Resposta bruta do Gemini: {response.text}")
        # Tentar uma correção mais agressiva ou retornar um dicionário vazio com erro
        return {"Error": "Failed to parse Gemini JSON output", "RawGeminiOutput": response.text}


# Função para gerar embeddings
def get_text_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos usando o modelo de embedding do Vertex AI.
    """
    model = TextEmbeddingModel.from_pretrained("text-embedding-004") # Ou outro modelo de embedding adequado
    embeddings = model.embed(texts)
    return [embedding.values for embedding in embeddings]

# Função para normalizar termos (ex: Google Workspace, GWS)
# Pode ser expandida com um dicionário de sinônimos ou com a ajuda de um LLM
def normalize_term(term: str) -> str:
    term = term.lower().strip()
    replacements = {
        "google workspace": "google workspace",
        "gws": "google workspace",
        "contas google": "google workspace contas",
        "google cloud platform": "google cloud platform",
        "gcp": "google cloud platform",
        "armazenamento em nuvem": "cloud storage",
        "nuvem pública": "cloud public",
        "ia": "inteligência artificial",
        "inteligência artificial": "inteligência artificial",
        "robô": "robô",
        "robotizado": "robô",
        "chatbots": "chatbot",
        "ura": "unidade de resposta audível",
        "geração de linguagem natural": "geração de linguagem natural"
    }
    # Tenta encontrar correspondência exata primeiro, depois parcial
    for key, value in replacements.items():
        if key in term:
            return value
    return term

# Função para cruzar ativos com requisitos
def cross_reference_assets(extracted_requirements: dict, assets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza os requisitos extraídos do edital com os ativos da planilha.
    Retorna um DataFrame com a análise de atendimento.
    """
    if assets_df.empty:
        return pd.DataFrame(columns=['Requisito', 'Tipo', 'Status', 'Evidência', 'Ação Necessária'])

    # Extrair os requisitos relevantes para cruzamento
    all_requirements = []
    
    # Requisitos do Objeto/Qualificação Técnica Específica
    for req in extracted_requirements.get("RequisitosObjetoQualificacaoTecnicaEspecifica", []):
        all_requirements.append(req.get("Descricao", "") + " " + " ".join(req.get("Detalhes", [])))
    
    # Requisitos de Habilitação Técnica Geral (se houver menções específicas de tecnologias/serviços)
    for req_type in ["Juridica", "Fiscal", "EconomicoFinanceira", "TecnicaGeral"]:
        for req_item in extracted_requirements.get("RequisitosHabilitacao", {}).get(req_type, []):
            # Filtra itens genéricos de habilitação e foca nos técnicos/de serviço
            if "atestado" in req_item.lower() or "certificação" in req_item.lower() or "serviço especializado" in req_item.lower():
                all_requirements.append(req_item)
    
    all_requirements = [req for req in all_requirements if req.strip()] # Remove vazios

    if not all_requirements:
        return pd.DataFrame(columns=['Requisito', 'Tipo', 'Status', 'Evidência', 'Ação Necessária'])

    # Prepare os textos para embeddings (ativos e requisitos)
    assets_texts = assets_df['ProdutosConcatenados'].fillna('') + ' ' + assets_df['Resumo_Objeto_Consolidado'].fillna('')
    assets_embeddings = get_text_embeddings(assets_texts.tolist())
    
    requirements_embeddings = get_text_embeddings(all_requirements)

    analysis_results = []

    for i, req_text in enumerate(all_requirements):
        req_embedding = requirements_embeddings[i]
        
        best_match_score = -1
        best_match_asset_index = -1
        
        # Calcule a similaridade do requisito com todos os ativos
        if assets_embeddings and req_embedding: # Garante que não está vazio
            # Reshape para cosine_similarity: (1, n_features) para o vetor único, (n_samples, n_features) para múltiplos vetores
            similarities = cosine_similarity(np.array(req_embedding).reshape(1, -1), np.array(assets_embeddings)).flatten()
            best_match_asset_index = np.argmax(similarities)
            best_match_score = similarities[best_match_asset_index]

        status = "🚨 Não atende / Bloqueador"
        evidence = "—"
        action_needed = "Buscar solução ou impugnar"

        if best_match_score >= 0.8:  # Limiar de alta similaridade
            matched_asset = assets_df.iloc[best_match_asset_index]
            
            # Lógica de Classificação (refinada)
            # Exemplo: verificação de quantitativos e menções de IA/GCP
            if "quantitativo mínimo" in req_text.lower():
                # Tente extrair quantitativos do requisito e do ativo para comparação
                # Isso exigiria uma lógica de NLP mais avançada para parsear "1.000.000 unidades de consumo" etc.
                pass # Lógica complexa, deixada como placeholder
            
            # Normalizar termos para verificar menções de tecnologia/IA
            normalized_req = normalize_term(req_text)
            normalized_asset_products = normalize_term(matched_asset['ProdutosConcatenados'].fillna(''))
            normalized_asset_summary = normalize_term(matched_asset['Resumo_Objeto_Consolidado'].fillna(''))
            
            # Verificar se a tecnologia/serviço principal do requisito está no ativo
            tech_match = False
            if "google cloud platform" in normalized_req and ("google cloud platform" in normalized_asset_products or "google cloud platform" in normalized_asset_summary):
                tech_match = True
            elif "google workspace" in normalized_req and ("google workspace" in normalized_asset_products or "google workspace" in normalized_asset_summary):
                tech_match = True
            elif "robô" in normalized_req and ("robô" in normalized_asset_products or "robô" in normalized_asset_summary):
                tech_match = True
            elif "inteligência artificial" in normalized_req and ("inteligência artificial" in normalized_asset_products or "inteligência artificial" in normalized_asset_summary):
                tech_match = True
            
            
            # Verificação de Certificações e IA (exemplo, precisa de mapeamento real)
            # 'Certificacoes_Valores_Mencoes_IA' no seu Google Sheet
            asset_certifications_ia = str(matched_asset.get('Certificacoes_Valores_Mencoes_IA', '')).lower()
            
            ia_mentioned_in_asset = "ia" in normalized_asset_products or "ia" in normalized_asset_summary or "inteligência artificial" in asset_certifications_ia
            
            if tech_match:
                if (matched_asset['Tipo_Contrato'].lower() == 'contrato' and "atestado" not in req_text.lower()) or \
                   (matched_asset['Tipo_Contrato'].lower() == 'atestado' and "atestado" in req_text.lower()):
                    status = "✅ Atende diretamente"
                    evidence = f"{matched_asset['Tipo_Contrato']} - {matched_asset['Nome_Orgao']} - {matched_asset['Ano_Contrato']}"
                    action_needed = "Nenhuma"
                elif matched_asset['Tipo_Contrato'].lower() in ['contrato', 'sow'] and "sow" in matched_asset['ProdutosConcatenados'].lower(): # Exemplo de "indireto"
                    status = "⚠️ Atende indiretamente (combinando contrato e SOW)"
                    evidence = f"{matched_asset['Tipo_Contrato']} + SOW - {matched_asset['Nome_Orgao']} - {matched_asset['Ano_Contrato']}"
                    action_needed = "Detalhar no recurso"
            
            # Refinamento para IA, se o requisito for explicitamente sobre IA e o ativo mencionar IA
            if "inteligência artificial" in normalized_req or "ia" in normalized_req:
                if ia_mentioned_in_asset:
                    status = "✅ Atende diretamente" # Pode ser mais granular se necessário
                    evidence = f"{matched_asset['Tipo_Contrato']} - {matched_asset['Nome_Orgao']} - {matched_asset['Ano_Contrato']} (com IA)"
                    action_needed = "Nenhuma"
                else:
                    # Se o requisito é de IA mas o ativo correspondente não menciona IA explicitamente
                    if status != "✅ Atende diretamente": # Não sobrescreve se já atende diretamente por outro motivo
                        status = "🚨 Não atende / Bloqueador (Requisito de IA não comprovado no ativo)"
                        action_needed = "Buscar evidência específica de IA ou desenvolver"


        analysis_results.append({
            "Requisito": req_text,
            "Tipo": "Objeto/Qualificação Técnica" if "RequisitosObjetoQualificacaoTecnicaEspecifica" in json.dumps(extracted_requirements) else "Habilitação Técnica",
            "Status": status,
            "Evidência": evidence,
            "Ação Necessária": action_needed
        })

    return pd.DataFrame(analysis_results)

# Exemplo de uso (para teste local)
if __name__ == "__main__":
    from config import GOOGLE_CLOUD_PROJECT_ID, GOOGLE_CLOUD_LOCATION, GOOGLE_SHEET_URL, GOOGLE_SHEET_TAB_NAME, SERVICE_ACCOUNT_FILE
    from google_sheets_integrator import get_google_sheet_data
    import os

    # Inicializar Vertex AI
    initialize_vertex_ai(GOOGLE_CLOUD_PROJECT_ID, GOOGLE_CLOUD_LOCATION)

    # Simular texto do edital (use um trecho relevante dos seus PDFs)
    sample_edital_text = """
    2.1. Registro de preço para contratação de empresa especializada em fornecimento
 de créditos e em prestação de serviços de suporte técnico continuado,
 desenvolvimento e manutenção de soluções e transferência de conhecimento sob
 demanda, na nuvem pública Google Cloud Plataform (GCP), pelo período de 36 (trinta
 e seis) meses, conforme especificações constantes no Edital do Pregão Eletrônico nº
 04/2024 e seus anexos.
 6.1.3 Qualificação técnica
 6.1.3.1. A licitante deverá apresentar Atestado(s) de capacidade técnica de
 fornecido(s) por pessoa jurídica de direito público ou privado, que comprove(m) que a
 licitante já forneceu o serviço especializado no fornecimento de créditos de nuvem
 pública Google Cloud Plataform (GCP) pelo período mínimo de 12 meses.
 6.1.3.2. Para fins da comprovação de aptidão para execução de serviço de
 complexidade tecnológica e operacional equivalente ou superior com o objeto desta
 contratação, os atestados deverão dizer respeito a contratos executados com vigência
 mínima de doze meses, por meio da apresentação de certidões ou atestados, por
 pessoas jurídicas de direito público ou privado, com as seguintes características
 mínimas:
 a) Demonstração de provimento de subscrições de serviços Google Cloud;
 b) Demonstração de execução de pelo menos um serviço especializado na
 plataforma Google Cloud Platform,
 c) O gerenciamento e a operação de instâncias de máquinas virtuais e de
 instâncias de banco de dados em ambiente de nuvem pública Google
 Cloud;
 g) Migração de pelo menos 1 (um) banco de dados legado, de ambiente on-
 premise, para um banco de dados gerenciado nativo de um provedor de
 nuvem pública.
 h) Volume mínimo de 1.000.000 (um milhão) de unidades de consumo em
 nuvem ou moeda equivalente (R$, USN, CSN, USIN).
 Deverá ser
 comprovada execução do montante mínimo informado, sendo vetadas
 apresentações apenas de valores contratuais totais.
 6.1.5. Caso não seja o fabricante, a LICITANTE deverá apresentar documento
 que comprove estar autorizada e credenciada a comercializar os produtos
 disponíveis da Google Cloud Plataform (GCP), sendo um "premier partner".
 6.1.6. Declaração que detém a quantidade mínima e profissionais em seu quadro
 ou prestadores de serviço certificados com o objetivo de garantir o mínimo de
 qualidade na prestação de serviços em relação ao provedor de nuvem
 oferecido (comprovada na assinatura do contrato): no mínimo 2 (dois)
 profissionais com certificação de arquitetura em nuvem;
    """
    
    # Simular dados de ativos da planilha (para testes sem acessar a API real repetidamente)
    # Em um ambiente real, você carregaria isso de google_sheets_integrator
    try:
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            ativos_df_test = get_google_sheet_data(GOOGLE_SHEET_URL, GOOGLE_SHEET_TAB_NAME, SERVICE_ACCOUNT_FILE)
        else:
            print("AVISO: SERVICE_ACCOUNT_FILE não encontrado. Usando DataFrame de teste mock.")
            # Mock de dados se o arquivo de serviço não estiver disponível
            ativos_data = {
                'ID': [1, 2, 3],
                'Tipo_Contrato': ['Atestado', 'Contrato', 'Certificação'],
                'Nome_Orgao': ['MPPE', 'TJES', 'Empresa X'],
                'Ano_Contrato': [2024, 2023, 2022],
                'Resumo_Objeto_Consolidado': [
                    'Fornecimento de 1.200.000 unidades de consumo Google Cloud Platform (GCP) com suporte técnico e migração de banco de dados.',
                    'Prestação de serviços de desenvolvimento de software em nuvem, incluindo uso de IA para otimização de processos.',
                    'Certificação ISO 27001 e parceria Google Cloud Premier Partner.'
                ],
                'ProdutosConcatenados': [
                    'Google Cloud Platform, GCP Storage, Google Kubernetes Engine, BigQuery, 1.2M unidades',
                    'Desenvolvimento de Chatbot com IA, Machine Learning, APIs, Google Workspace',
                    'ISO 27001, Google Cloud Premier Partner'
                ],
                'Certificacoes_Valores_Mencoes_IA': [
                    'Certificação GCP Professional Cloud Architect, Menção de IA em projetos',
                    'Menção de IA em SOW, Certificação Scrum Master',
                    'Certificação ISO 27001, Google Cloud Premier Partner'
                ]
            }
            ativos_df_test = pd.DataFrame(ativos_data)


    except Exception as e:
        print(f"Erro ao carregar dados da planilha para teste de ai_analyzer: {e}")
        ativos_df_test = pd.DataFrame() # DataFrame vazio para evitar quebra

    print("--- EXTRAINDO REQUISITOS COM GEMINI ---")
    extracted_reqs = extract_requirements_with_gemini(sample_edital_text)
    print(json.dumps(extracted_reqs, indent=2, ensure_ascii=False))

    print("\n--- CRUZANDO ATIVOS COM REQUISITOS ---")
    analysis_df = cross_reference_assets(extracted_reqs, ativos_df_test)
    print(analysis_df.to_markdown(index=False))