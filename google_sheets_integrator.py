# google_sheets_integrator.py
import gspread
import pandas as pd
import os
# from google.oauth2 import service_account # Não é mais explicitamente necessário aqui se gspread gerenciar

def get_google_sheet_data(sheet_url: str, tab_name: str) -> pd.DataFrame:
    """
    Carrega dados de uma aba específica de uma planilha do Google Sheets.
    Assume que a primeira linha contém os nomes das colunas.
    No Cloud Run, usará as credenciais da conta de serviço anexada ao serviço.
    """
    try:
        # Autenticação: gspread tentará usar as credenciais da conta de serviço do ambiente
        # Certifique-se de que a conta de serviço do Cloud Run tem permissão de Sheets Reader/Editor
        gc = gspread.service_account() # NENHUM ARQUIVO DE CHAVE AQUI!
        
        spreadsheet = gc.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet(tab_name)
        
        data = worksheet.get_all_records() # Retorna uma lista de dicionários
        df = pd.DataFrame(data)
        
        # Opcional: Limpar colunas totalmente vazias ou linhas com todos os valores vazios se necessário
        df.dropna(axis=1, how='all', inplace=True) # Remove colunas totalmente vazias
        df.dropna(axis=0, how='all', inplace=True) # Remove linhas totalmente vazias
        
        print(f"Dados carregados da planilha: {df.shape[0]} linhas, {df.shape[1]} colunas.")
        return df

    except gspread.exceptions.SpreadsheetNotFound:
        raise Exception(f"Planilha não encontrada na URL: {sheet_url}")
    except gspread.exceptions.WorksheetNotFound:
        raise Exception(f"Aba '{tab_name}' não encontrada na planilha.")
    except Exception as e:
        print(f"Erro ao carregar dados da planilha: {e}")
        raise

# Exemplo de uso (para teste local)
if __name__ == "__main__":
    from config import GOOGLE_SHEET_URL, GOOGLE_SHEET_TAB_NAME
    
    # Para testar LOCALMENTE SEM ARQUIVO DE CHAVE:
    # Você precisaria usar 'gcloud auth application-default login'
    # e ter as permissões necessárias para sua conta de usuário na planilha.
    # Ou, para testar a funcionalidade com um mock:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"): # Se estiver usando um arquivo de chave local
        # Isso é um fallback para teste local com arquivo, mas não é usado em Cloud Run
        gc = gspread.service_account(filename=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        # Resto do código para testar com o arquivo
    else:
        print("Para testar google_sheets_integrator.py localmente sem arquivo de chave,")
        print("assegure-se de que 'gcloud auth application-default login' foi executado")
        print("e que sua conta de usuário tem acesso à planilha.")
        print("Tentando carregar dados sem arquivo de chave...")
        try:
            ativos_df = get_google_sheet_data(GOOGLE_SHEET_URL, GOOGLE_SHEET_TAB_NAME)
            print("--- DADOS DA PLANILHA ---")
            print(ativos_df.head())
            print("--- FIM DOS DADOS ---")
        except Exception as e:
            print(f"Falha ao carregar dados da planilha para teste: {e}")