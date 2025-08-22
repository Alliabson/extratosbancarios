import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict, Any
import unicodedata
import json
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import io
import traceback
from PIL import Image
import time

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide", page_title="Analisador de Extratos Bancários")
st.title("📊 Analisador de Extratos Bancários com IA")
st.write("Faça o upload dos seus extratos em PDF. A análise será feita por regras e, se necessário, pela IA do Gemini.")

# --- FUNÇÕES DE LÓGICA DE ANÁLISE ---

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extrai texto de TODAS as páginas de um arquivo PDF, preservando as quebras de linha."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            num_pages = len(doc)
            st.sidebar.info(f"PDF possui {num_pages} página(s)")
            
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                full_text += f"\n{page_text}"
                
                if len(page_text.strip()) > 0:
                    st.sidebar.write(f"Página {page_num}: {len(page_text)} caracteres extraídos")
                else:
                    st.sidebar.warning(f"Página {page_num}: Nenhum texto extraído")
                
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return ""
    
    if len(full_text.strip()) > 0:
        st.sidebar.info(f"Total de texto extraído: {len(full_text)} caracteres")
    else:
        st.sidebar.error("Nenhum texto pôde ser extraído do PDF.")
    
    return full_text

def parse_amount(amount_str: str) -> float:
    """Converte uma string de valor monetário para float."""
    if not isinstance(amount_str, str):
        return 0.0
    
    cleaned_str = re.sub(r'[^\d,\-\.]', '', str(amount_str))
    
    is_negative = '-' in cleaned_str
    cleaned_str = cleaned_str.replace('-', '')
    
    if '.' in cleaned_str and ',' in cleaned_str:
        cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
    elif ',' in cleaned_str:
        cleaned_str = cleaned_str.replace(',', '.')
    
    try:
        value = float(cleaned_str)
        return -value if is_negative else value
    except (ValueError, TypeError):
        return 0.0

def parse_date(date_str: str) -> datetime:
    """Tenta parsear uma string de data em vários formatos diferentes."""
    if not isinstance(date_str, str):
        return pd.NaT
    
    date_str = date_str.strip()
    
    # Remove pontos e espaços extras
    date_str = re.sub(r'\.', '', date_str)
    date_str = re.sub(r'\s+', ' ', date_str)
    
    # Mapa de meses em português
    month_map = {
        'jan': '01', 'jan.': '01', 'janeiro': '01',
        'fev': '02', 'fev.': '02', 'fevereiro': '02',
        'mar': '03', 'mar.': '03', 'março': '03',
        'abr': '04', 'abr.': '04', 'abril': '04',
        'mai': '05', 'mai.': '05', 'maio': '05',
        'jun': '06', 'jun.': '06', 'junho': '06',
        'jul': '07', 'jul.': '07', 'julho': '07',
        'ago': '08', 'ago.': '08', 'agosto': '08',
        'set': '09', 'set.': '09', 'setembro': '09',
        'out': '10', 'out.': '10', 'outubro': '10',
        'nov': '11', 'nov.': '11', 'novembro': '11',
        'dez': '12', 'dez.': '12', 'dezembro': '12'
    }
    
    # Tenta diferentes formatos de data
    formats_to_try = [
        # Formato DD/MM/AAAA
        r'(\d{2})/(\d{2})/(\d{4})',
        # Formato DD-MM-AAAA
        r'(\d{2})-(\d{2})-(\d{4})',
        # Formato DD.MM.AAAA
        r'(\d{2})\.(\d{2})\.(\d{4})',
        # Formato AAAA/MM/DD
        r'(\d{4})/(\d{2})/(\d{2})',
        # Formato DD de Mês de AAAA
        r'(\d{1,2}) de ([a-zA-Zç]+) de (\d{4})',
        r'(\d{1,2}) de ([a-zA-Zç]+)\.? de (\d{4})',
        # Formato Mês/AAAA
        r'([a-zA-Zç]+)/(\d{4})',
        # Formato MM/AAAA
        r'(\d{2})/(\d{4})',
    ]
    
    for pattern in formats_to_try:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                if 'de ' in pattern:  # Formato com texto (ex: "01 de Janeiro de 2025")
                    day = match.group(1).zfill(2)
                    month_str = match.group(2).lower()
                    year = match.group(3)
                    
                    # Traduz o mês em português para número
                    for month_name, month_num in month_map.items():
                        if month_name in month_str:
                            month = month_num
                            break
                    else:
                        continue  # Mês não reconhecido
                    
                    date_str_parsed = f"{day}/{month}/{year}"
                    return pd.to_datetime(date_str_parsed, format='%d/%m/%Y')
                
                elif len(match.groups()) == 3:  # Formato com dia, mês e ano
                    if pattern == r'(\d{4})/(\d{2})/(\d{2})':  # AAAA/MM/DD
                        year, month, day = match.groups()
                    else:  # DD/MM/AAAA ou similar
                        day, month, year = match.groups()
                    
                    return pd.to_datetime(f"{day.zfill(2)}/{month.zfill(2)}/{year}", format='%d/%m/%Y')
                
                elif len(match.groups()) == 2:  # Formato apenas com mês e ano
                    if pattern == r'([a-zA-Zç]+)/(\d{4})':  # Mês/AAAA
                        month_str, year = match.groups()
                        # Traduz o mês em português para número
                        for month_name, month_num in month_map.items():
                            if month_name in month_str.lower():
                                month = month_num
                                break
                        else:
                            continue  # Mês não reconhecido
                    else:  # MM/AAAA
                        month, year = match.groups()
                    
                    # Usa o primeiro dia do mês como data padrão
                    return pd.to_datetime(f"01/{month.zfill(2)}/{year}", format='%d/%m/%Y')
                    
            except (ValueError, Exception):
                continue
    
    return pd.NaT

def parse_itau(text: str) -> List[Dict[str, Any]]:
    """Parser robusto para extratos do Itaú que analisa linha por linha."""
    transactions = []
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Padrão para transações do Itaú: data, descrição e valor no final
        pattern = r'^(\d{2}/\d{2}/\d{4})\s+(.*?)\s+(-?[\d\.]*,\d{2})$'
        match = re.search(pattern, line)
        
        if match:
            date_str = match.group(1)
            description = match.group(2).strip()
            amount_str = match.group(3)
            
            # Ignora linhas de saldo
            if any(term in description.upper() for term in ['SALDO', 'LANÇAMENTOS', 'EXTRATO', 'SALDO ANTERIOR', 'SALDO DO DIA']):
                continue
            
            parsed_date = parse_date(date_str)
            if pd.isna(parsed_date):
                continue
                
            transactions.append({
                "date": parsed_date,
                "description": description,
                "amount": parse_amount(amount_str)
            })
    
    return transactions

def parse_santander(text: str) -> List[Dict[str, Any]]:
    """Parser para extratos do Santander."""
    transactions = []
    
    lines = text.split('\n')
    
    # Padrão mais flexível para Santander
    date_patterns = [
        r'^\d{2}/\d{2}/\d{4}',  # DD/MM/AAAA
        r'^\d{2}-\d{2}-\d{4}',   # DD-MM-AAAA
        r'^\d{2} de [a-zA-Zç]+ de \d{4}',  # DD de Mês de AAAA
        r'^\d{1,2}/\d{4}',       # MM/AAAA
    ]
    
    amount_pattern = r'(-?\d{1,3}(?:\.\d{3})*,\d{2})$'
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Verifica se a linha começa com uma data em qualquer formato
        date_match = None
        for pattern in date_patterns:
            date_match = re.search(pattern, line)
            if date_match:
                break
        
        if date_match:
            date_str = date_match.group(0)
            
            # Encontra o valor no final da linha
            amount_match = re.search(amount_pattern, line)
            if amount_match:
                amount_str = amount_match.group(1)
                description = line[len(date_str):-len(amount_str)].strip()
                
                # Ignora linhas de saldo
                if any(term in description.upper() for term in ['SALDO', 'S A L D O', 'EXTRATO', 'SALDO ANTERIOR', 'SALDO DO DIA']):
                    continue
                
                parsed_date = parse_date(date_str)
                if pd.isna(parsed_date):
                    continue
                    
                transactions.append({
                    "date": parsed_date,
                    "description": description,
                    "amount": parse_amount(amount_str)
                })
    
    return transactions

def parse_inter(text: str) -> List[Dict[str, Any]]:
    """Parser para extratos do Banco Inter."""
    transactions = []
    current_date = None
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Verifica se é uma linha de data em vários formatos
        date_patterns = [
            r'(\d{1,2} de [a-zA-Zç]+ de \d{4})',
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{2}-\d{2}-\d{4})',
        ]
        
        date_match = None
        for pattern in date_patterns:
            date_match = re.search(pattern, line, re.IGNORECASE)
            if date_match:
                break
        
        if date_match:
            date_str = date_match.group(1)
            try:
                parsed_date = parse_date(date_str)
                if not pd.isna(parsed_date):
                    current_date = parsed_date
            except ValueError:
                continue
        elif current_date:
            # Verifica se a linha contém um valor monetário
            amount_match = re.search(r'R\$\s*([\d\.]+,\d{2})', line)
            if amount_match:
                amount_str = amount_match.group(1)
                description = line.replace(f"R$ {amount_str}", "").strip()
                
                if any(term in description.upper() for term in ['SALDO', 'EXTRATO', 'SALDO ANTERIOR', 'SALDO DO DIA']):
                    continue
                
                transactions.append({
                    "date": current_date,
                    "description": description,
                    "amount": parse_amount(amount_str)
                })
    
    return transactions

def safe_json_parse(json_str):
    """Tenta analisar JSON de forma segura."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        return []

def parse_with_gemini(text: str, api_key: str) -> List[Dict[str, Any]]:
    """Usa a API do Gemini para extrair transações."""
    if not api_key:
        return []
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        text_sample = text[:8000] if len(text) > 8000 else text
        
        prompt = f"""
        Extraia transações bancárias do texto abaixo. Retorne APENAS JSON com:
        - date: DD/MM/AAAA
        - description: texto completo
        - amount: número (negativo para débitos)
        
        Texto: {text_sample}
        
        Exemplo: [{{"date": "01/01/2025", "description": "PAGAMENTO", "amount": -100.00}}]
        """
        
        try:
            response = model.generate_content(prompt)
            cleaned_response = response.text.strip()
            
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            transactions_json = safe_json_parse(cleaned_response)
            
            transactions = []
            for t in transactions_json:
                try:
                    parsed_date = parse_date(t['date'])
                    if not pd.isna(parsed_date):
                        transactions.append({
                            "date": parsed_date,
                            "description": t['description'],
                            "amount": float(t['amount'])
                        })
                except:
                    continue
            
            return transactions
            
        except Exception:
            return []
        
    except Exception:
        return []

def detect_bank_and_parse(text: str, filename: str, gemini_key: str) -> List[Dict[str, Any]]:
    """Detecta o banco e faz o parsing."""
    normalized_text = unicodedata.normalize('NFKD', text.lower())
    normalized_text = "".join([c for c in normalized_text if not unicodedata.combining(c)])
    
    parser = None
    bank_name = "Desconhecido"
    
    if 'itau' in normalized_text:
        bank_name = "Itaú"
        parser = parse_itau
    elif 'santander' in normalized_text:
        bank_name = "Santander"
        parser = parse_santander
    elif 'inter' in normalized_text:
        bank_name = "Banco Inter"
        parser = parse_inter
    
    st.sidebar.info(f"Arquivo '{filename}' identificado como: {bank_name}")
    
    transactions = []
    if parser:
        transactions = parser(text)
        st.sidebar.info(f"Parser encontrou {len(transactions)} transações")

    if (len(transactions) < 5 and gemini_key) or (not parser and gemini_key):
        st.sidebar.warning("Usando Gemini AI...")
        gemini_transactions = parse_with_gemini(text, gemini_key)
        if gemini_transactions:
            transactions = gemini_transactions
            st.sidebar.info(f"Gemini encontrou {len(transactions)} transações")
    
    return transactions

def categorize_transaction(description: str) -> str:
    """Categoriza transações."""
    if not isinstance(description, str):
        return 'Outros'
        
    desc_lower = description.lower()
    categories = {
        'Receitas': ['pix recebido', 'salário', 'deposito', 'transferencia recebida'],
        'Alimentação': ['ifood', 'restaurante', 'mercado', 'supermercado'],
        'Moradia': ['aluguel', 'condominio', 'água', 'luz', 'energia'],
        'Transporte': ['uber', 'posto', 'gasolina', 'estacionamento'],
        'Compras': ['shopping', 'lojas', 'mercado pag'],
        'Saúde': ['farmacia', 'drogaria', 'hospital', 'plano de saúde'],
        'Serviços': ['internet', 'telefone', 'streaming'],
    }
    
    for category, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return 'Outros'

# --- INTERFACE STREAMLIT ---

if 'excluded_ids' not in st.session_state:
    st.session_state.excluded_ids = set()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if "gemini_api_key" not in st.secrets:
    st.error("Configure a chave API do Gemini nos secrets.")
    st.stop()

gemini_api_key = st.secrets["gemini_api_key"]

with st.sidebar:
    st.header("Controles")
    st.success("API do Gemini configurada!")
    
    uploaded_files = st.file_uploader(
        "Selecione os PDFs",
        type="pdf",
        accept_multiple_files=True
    )
    
    filter_term = st.text_input("Filtrar por nome:")

if uploaded_files:
    current_filenames = [f.name for f in uploaded_files]
    
    if 'df_original' not in st.session_state or st.session_state.get('processed_files') != current_filenames:
        with st.spinner("Processando..."):
            all_transactions = []
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                text = extract_text_from_pdf(file_content)
                
                if not text or len(text.strip()) < 50:
                    st.error(f"Erro no arquivo {uploaded_file.name}")
                    continue
                
                transactions = detect_bank_and_parse(text, uploaded_file.name, gemini_api_key)
                all_transactions.extend(transactions)
                st.sidebar.success(f"{uploaded_file.name}: {len(transactions)} transações")

            if not all_transactions:
                st.error("Nenhuma transação encontrada.")
                st.stop()

            df = pd.DataFrame(all_transactions)
            df['category'] = df['description'].apply(categorize_transaction)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'id'}, inplace=True)
            
            st.session_state.df_original = df
            st.session_state.processed_files = current_filenames
            st.session_state.excluded_ids = set()

    df_processed = st.session_state.df_original.copy()
    
    if filter_term:
        mask = ~df_processed['description'].str.contains(filter_term, case=False)
        removed = df_processed[~mask]
        df_processed = df_processed[mask]
        if len(removed) > 0:
            st.sidebar.info(f"{len(removed)} transações removidas")

    if st.session_state.excluded_ids:
        df_processed = df_processed[~df_processed['id'].isin(st.session_state.excluded_ids)]

    st.header("Análise Financeira")

    if not df_processed.empty:
        total_income = df_processed[df_processed['amount'] > 0]['amount'].sum()
        total_expenses = df_processed[df_processed['amount'] < 0]['amount'].sum()
        net_balance = total_income + total_expenses
        
        # Cálculo correto dos meses
        df_processed['ano_mes'] = df_processed['date'].dt.to_period('M')
        unique_months = df_processed['ano_mes'].nunique()
        months_analyzed = max(unique_months, 1)
        
        # Mostra todos os meses encontrados
        st.sidebar.info(f"Meses detectados: {sorted(df_processed['ano_mes'].astype(str).unique())}")
        
        average_income = total_income / months_analyzed if months_analyzed > 0 else 0
        presumed_income = average_income * 0.30

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entradas", f"R$ {total_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col2.metric("Saídas", f"R$ {abs(total_expenses):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col3.metric("Saldo", f"R$ {net_balance:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col4.metric("Meses", months_analyzed)

        col5, col6 = st.columns(2)
        col5.metric("Média Mensal", f"R$ {average_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col6.metric("Capacidade 30%", f"R$ {presumed_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        
        # Análise por categoria
        st.subheader("Por Categoria")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Despesas**")
            expenses = df_processed[df_processed['amount'] < 0].groupby('category')['amount'].sum().abs()
            for cat, amount in expenses.items():
                st.write(f"{cat}: R$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        
        with col2:
            st.write("**Receitas**")
            income = df_processed[df_processed['amount'] > 0].groupby('category')['amount'].sum()
            for cat, amount in income.items():
                st.write(f"{cat}: R$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

    st.subheader("Resumo Mensal")
    if not df_processed.empty:
        monthly = df_processed.groupby('ano_mes').agg({
            'amount': [('Entradas', lambda x: x[x > 0].sum()), 
                      ('Saídas', lambda x: x[x < 0].sum()),
                      ('Saldo', 'sum')]
        }).round(2)
        
        monthly.columns = ['Entradas', 'Saídas', 'Saldo']
        monthly['Mês'] = monthly.index.astype(str)
        monthly = monthly[['Mês', 'Entradas', 'Saídas', 'Saldo']]
        
        for col in ['Entradas', 'Saídas', 'Saldo']:
            monthly[col] = monthly[col].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        
        st.dataframe(monthly, use_container_width=True, hide_index=True)

    st.subheader("Todas as Transações")
    if not df_processed.empty:
        df_display = df_processed.copy()
        df_display['Data'] = df_display['date'].dt.strftime('%d/%m/%Y')
        df_display['Valor'] = df_display['amount'].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        
        st.dataframe(
            df_display[['Data', 'description', 'Valor', 'category']].rename(
                columns={'description': 'Descrição', 'category': 'Categoria'}
            ),
            use_container_width=True,
            hide_index=True,
            height=min(800, 35 * len(df_display) + 38)
        )

else:
    st.info("Aguardando upload de PDFs.")
