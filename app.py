import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import unicodedata
import json
import google.generativeai as genai
from io import BytesIO
import traceback

# --- CONFIGURAÇÕES GERAIS ---
MAX_PAGES_TO_PROCESS = 30  # Limite para evitar processamento muito longo
MAX_TEXT_LENGTH_FOR_AI = 30000  # Limite de caracteres para envio ao Gemini

# --- FUNÇÕES DE LÓGICA DE ANÁLISE ---

@st.cache_data
def extract_text_from_pdf(file_content: bytes, max_pages: int = MAX_PAGES_TO_PROCESS) -> str:
    """Extrai texto de um arquivo PDF de forma robusta, com tratamento de erros."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                if page_num >= max_pages:
                    st.warning(f"Documento muito grande. Processando apenas as primeiras {max_pages} páginas.")
                    break
                full_text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {str(e)}")
        st.error(traceback.format_exc())
    return full_text

def parse_amount(amount_str: str) -> float:
    """Converte uma string de valor monetário para float de forma segura."""
    if not isinstance(amount_str, str):
        return 0.0
    
    # Remove caracteres não numéricos exceto , e .
    cleaned_str = re.sub(r'[^\d,-.]', '', amount_str.replace('R$', '').strip())
    
    # Verifica se está no formato brasileiro (1.234,56)
    if ',' in cleaned_str and '.' in cleaned_str:
        if cleaned_str.find(',') > cleaned_str.find('.'):  # Formato 1,234.56 (internacional)
            cleaned_str = cleaned_str.replace(',', '')
        else:  # Formato brasileiro 1.234,56
            cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
    elif ',' in cleaned_str:  # Apenas vírgula como separador decimal
        cleaned_str = cleaned_str.replace(',', '.')
    
    try:
        return float(cleaned_str)
    except (ValueError, TypeError):
        return 0.0

def parse_itau(text: str) -> List[Dict[str, Any]]:
    """Parser robusto para extratos do Itaú com tratamento de erros."""
    transactions = []
    lines_processed = 0
    
    # Regex melhorada para capturar datas e valores
    transaction_regex = re.compile(
        r'^(\d{2}/\d{2}/\d{4})'  # Data
        r'(.*?)'  # Descrição (não guloso)
        r'(-?\s*\d{1,3}(?:\.?\d{3})*(?:,\d{2})?)\s*$'  # Valor
    )
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = transaction_regex.search(line)
        if match:
            date_str, description, amount_str = match.groups()
            
            # Filtra linhas de cabeçalho/saldo
            if any(word in description.upper() for word in ['SALDO', 'LANÇAMENTOS', 'EXTRATO']):
                continue
                
            try:
                transactions.append({
                    "date": pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce'),
                    "description": description.strip(),
                    "amount": parse_amount(amount_str)
                })
                lines_processed += 1
            except Exception as e:
                st.warning(f"Erro ao processar linha: {line}\nErro: {str(e)}")
    
    return transactions

def parse_inter(text: str) -> List[Dict[str, Any]]:
    """Parser para extratos do Banco Inter com tratamento de erros."""
    transactions = []
    current_date = None
    date_header_regex = re.compile(r'(\d{1,2}\s+de\s+[A-Za-zç]+\s+de\s+\d{4})', re.IGNORECASE)
    
    month_map = {
        'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 
        'abril': 'April', 'maio': 'May', 'junho': 'June',
        'julho': 'July', 'agosto': 'August', 'setembro': 'September',
        'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'
    }
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Verifica se é um cabeçalho de data
        date_match = date_header_regex.search(line)
        if date_match:
            date_str = date_match.group(1)
            for pt, en in month_map.items():
                date_str = date_str.replace(pt, en)
            try:
                current_date = datetime.strptime(date_str, '%d de %B de %Y')
            except ValueError:
                continue
        elif current_date:
            # Verifica se a linha contém um valor (R$ no final)
            parts = line.split()
            if len(parts) > 1 and ('R$' in parts[-1] or 'RS' in parts[-1]):
                amount_str = parts[-1]
                description = " ".join(parts[:-1])
                
                if any(word in description.upper() for word in ['SALDO', 'RESUMO', 'TOTAL']):
                    continue
                    
                try:
                    transactions.append({
                        "date": current_date,
                        "description": description.strip(),
                        "amount": parse_amount(amount_str)
                    })
                except Exception as e:
                    st.warning(f"Erro ao processar linha: {line}\nErro: {str(e)}")
    
    return transactions

def parse_with_gemini(text: str, api_key: str) -> List[Dict[str, Any]]:
    """Usa a API do Gemini para extrair transações de forma robusta."""
    if not api_key:
        st.error("Chave de API do Gemini não fornecida.")
        return []
    
    try:
        # Limita o tamanho do texto para evitar erros
        text = text[:MAX_TEXT_LENGTH_FOR_AI]
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """Extraia transações bancárias do texto abaixo. Retorne APENAS um array JSON válido.
Cada transação deve ter: date (DD/MM/AAAA), description (string) e amount (número com sinal).
Exemplo válido:
[
  {"date": "01/06/2024", "description": "PIX ENVIADO", "amount": -150.00},
  {"date": "02/06/2024", "description": "DEPOSITO", "amount": 1000.00}
]

Texto do extrato:
"""
        prompt += text[:MAX_TEXT_LENGTH_FOR_AI]  # Garante que não exceda o limite
        
        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        
        # Limpeza robusta da resposta
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        raw_response = raw_response.strip()
        
        # Verifica se parece ser um JSON válido
        if not raw_response.startswith('[') or not raw_response.endswith(']'):
            st.error(f"Resposta do Gemini em formato inesperado. Início: {raw_response[:100]}...")
            return []
            
        # Parse do JSON com tratamento de erros
        try:
            transactions = json.loads(raw_response)
        except json.JSONDecodeError as e:
            st.error(f"Erro ao decodificar JSON: {str(e)}")
            st.error(f"Trecho problemático: {raw_response[max(0, e.pos-50):e.pos+50]}")
            return []
            
        # Validação e conversão dos dados
        valid_transactions = []
        for t in transactions:
            try:
                if not all(key in t for key in ['date', 'description', 'amount']):
                    continue
                    
                valid_transactions.append({
                    "date": pd.to_datetime(t['date'], format='%d/%m/%Y', errors='coerce'),
                    "description": str(t['description']),
                    "amount": float(t['amount'])
                })
            except Exception as e:
                st.warning(f"Transação ignorada: {t}. Erro: {str(e)}")
                continue
                
        # Remove transações com datas inválidas
        valid_transactions = [t for t in valid_transactions if pd.notna(t['date'])]
        
        return valid_transactions
        
    except Exception as e:
        st.error(f"Erro na API do Gemini: {str(e)}")
        st.error(traceback.format_exc())
        return []

def detect_bank_and_parse(text: str, filename: str, gemini_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Detecta o banco e faz o parse com fallback para Gemini."""
    # Normaliza o texto para comparação
    nfkd_form = unicodedata.normalize('NFKD', text.lower())
    normalized_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    # Tenta identificar o banco
    bank_parsers = {
        'itau': parse_itau,
        'banco inter': parse_inter
    }
    
    selected_parser = None
    for bank_name, parser in bank_parsers.items():
        if bank_name in normalized_text:
            st.sidebar.info(f"Arquivo '{filename}' identificado como: {bank_name.title()}")
            selected_parser = parser
            break
    
    # Primeira tentativa com parser específico
    transactions = []
    if selected_parser:
        with st.spinner(f"Processando com parser {selected_parser.__name__}..."):
            transactions = selected_parser(text)
    
    # Fallback para Gemini se necessário
    if gemini_key and (not transactions or len(transactions) < 3):
        st.sidebar.warning("Usando análise com Gemini AI...")
        with st.spinner("Processando com IA..."):
            gemini_transactions = parse_with_gemini(text, gemini_key)
            if gemini_transactions:
                transactions = gemini_transactions
            else:
                st.error("Gemini não retornou transações válidas.")
    
    return transactions

def categorize_transaction(description: str) -> str:
    """Categoriza transações com palavras-chave atualizadas."""
    desc_lower = description.lower()
    categories = {
        'Receitas': ['pix recebido', 'depósito', 'salário', 'rendimento', 'credito'],
        'Alimentação': ['ifood', 'restaurante', 'mercado', 'supermercado', 'padaria'],
        'Moradia': ['aluguel', 'condomínio', 'luz', 'água', 'energia', 'internet'],
        'Transporte': ['uber', 'taxi', 'posto', 'combustível', 'pedágio'],
        'Compras': ['amazon', 'shopee', 'mercado livre', 'lojas', 'shopping'],
        'Saúde': ['farmacia', 'drogaria', 'plano de saúde', 'hospital'],
        'Lazer': ['cinema', 'netflix', 'spotify', 'parque', 'viagem'],
        'Serviços & Taxas': ['tarifa', 'juros', 'multa', 'boleto', 'anuidade'],
        'Investimentos': ['aplicação', 'resgate', 'tesouro', 'ações', 'fundo'],
        'Outros': []
    }
    
    for category, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return 'Outros'

# --- INTERFACE STREAMLIT ---

def main():
    st.set_page_config(layout="wide", page_title="Analisador de Extratos Bancários")
    st.title("📊 Analisador de Extratos Bancários com IA")
    
    # Inicialização do estado da sessão
    if 'df_original' not in st.session_state:
        st.session_state.df_original = pd.DataFrame()
    if 'excluded_ids' not in st.session_state:
        st.session_state.excluded_ids = set()
    
    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        gemini_api_key = st.text_input("Chave da API Gemini", type="password")
        uploaded_files = st.file_uploader("Selecione os PDFs", type="pdf", accept_multiple_files=True)
        filter_name = st.text_input("Filtrar por nome (opcional)")
        st.info("Dicas: Para melhores resultados, use extratos em formato texto (não imagem).")
    
    # Processamento dos arquivos
    if uploaded_files:
        current_files = [f.name for f in uploaded_files]
        
        # Verifica se precisa reprocessar
        if 'processed_files' not in st.session_state or st.session_state.processed_files != current_files:
            with st.spinner("Processando arquivos..."):
                all_transactions = []
                
                for uploaded_file in uploaded_files:
                    try:
                        file_content = uploaded_file.getvalue()
                        text = extract_text_from_pdf(file_content)
                        
                        if not text:
                            st.error(f"Arquivo {uploaded_file.name} não contém texto legível.")
                            continue
                            
                        transactions = detect_bank_and_parse(text, uploaded_file.name, gemini_api_key)
                        all_transactions.extend(transactions)
                        
                    except Exception as e:
                        st.error(f"Erro ao processar {uploaded_file.name}: {str(e)}")
                        continue
                
                if all_transactions:
                    df = pd.DataFrame(all_transactions)
                    df['category'] = df['description'].apply(categorize_transaction)
                    df['id'] = range(len(df))  # ID único para cada transação
                    
                    # Ordena por data
                    df = df.sort_values('date', ascending=False).reset_index(drop=True)
                    
                    st.session_state.df_original = df
                    st.session_state.processed_files = current_files
                    st.session_state.excluded_ids = set()
                else:
                    st.error("Nenhuma transação válida encontrada nos arquivos.")
                    return
        
        # Filtros
        df_processed = st.session_state.df_original.copy()
        
        if filter_name:
            mask = ~df_processed['description'].str.contains(filter_name, case=False)
            df_processed = df_processed[mask]
        
        if st.session_state.excluded_ids:
            df_processed = df_processed[~df_processed['id'].isin(st.session_state.excluded_ids)]
        
        # Métricas
        st.header("Resumo Financeiro")
        
        if not df_processed.empty:
            # Cálculos
            df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce')
            income = df_processed[df_processed['amount'] > 0]['amount'].sum()
            expenses = df_processed[df_processed['amount'] < 0]['amount'].sum()
            balance = income + expenses  # Soma porque expenses é negativo
            
            months = df_processed['date'].dt.to_period('M').nunique()
            avg_income = income / months if months > 0 else 0
            avg_expenses = abs(expenses) / months if months > 0 else 0
            
            # Exibição
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Entradas", f"R$ {income:,.2f}".replace('.', 'X').replace(',', '.').replace('X', ','))
            col2.metric("Total Saídas", f"R$ {abs(expenses):,.2f}".replace('.', 'X').replace(',', '.').replace('X', ','))
            col3.metric("Saldo", f"R$ {balance:,.2f}".replace('.', 'X').replace(',', '.').replace('X', ','))
            
            # Gráficos e análises
            st.subheader("Análise por Categoria")
            category_summary = df_processed[df_processed['amount'] < 0].groupby('category')['amount'].sum().sort_values()
            st.bar_chart(abs(category_summary))
            
            # Transações
            st.subheader("Transações Detalhadas")
            
            # Formatação para exibição
            display_df = df_processed.copy()
            display_df['Data'] = display_df['date'].dt.strftime('%d/%m/%Y')
            display_df['Valor'] = display_df['amount'].apply(lambda x: f"R$ {abs(x):,.2f}" if x < 0 else f"R$ {x:,.2f}")
            display_df['Tipo'] = display_df['amount'].apply(lambda x: "Saída" if x < 0 else "Entrada")
            
            # Editor de dados
            edited_df = st.data_editor(
                display_df[['Data', 'description', 'Valor', 'Tipo', 'category']].rename(columns={
                    'description': 'Descrição',
                    'category': 'Categoria'
                }),
                column_config={
                    "Categoria": st.column_config.SelectboxColumn(
                        "Categoria",
                        options=sorted(display_df['category'].unique())
                    )
                },
                use_container_width=True,
                hide_index=True,
                num_rows="fixed"
            )
            
            # Botão para atualizar categorias
            if st.button("Salvar Categorias"):
                # Mapeia as categorias editadas de volta para o DataFrame original
                category_mapping = edited_df.set_index('Descrição')['Categoria'].to_dict()
                st.session_state.df_original['category'] = st.session_state.df_original['description'].map(category_mapping)
                st.success("Categorias atualizadas!")
            
            # Exportação
            st.download_button(
                "Exportar para Excel",
                df_processed.to_excel(BytesIO(), index=False),
                "transacoes.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Nenhuma transação para exibir após os filtros aplicados.")
    else:
        st.info("Faça o upload de arquivos PDF para começar a análise.")

if __name__ == "__main__":
    main()
