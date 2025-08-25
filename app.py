import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict, Any
import unicodedata
import json
import google.generativeai as genai
import time
import traceback

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide", page_title="Analisador de Extratos Bancários")
st.title("📊 Analisador de Extratos Bancários com IA")
st.write("Faça o upload dos seus extratos em PDF. A análise será feita por regras e, se necessário, pela IA do Gemini.")

# --- FUNÇÕES DE LÓGICA DE ANÁLISE ---

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extrai texto de TODAS as páginas de um arquivo PDF."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            num_pages = len(doc)
            st.sidebar.info(f"PDF possui {num_pages} página(s)")
            
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                full_text += f"\n--- PAGE {page_num} ---\n{page_text}"
                
                if len(page_text.strip()) > 0:
                    st.sidebar.write(f"Página {page_num}: {len(page_text)} caracteres extraídos")
                else:
                    st.sidebar.warning(f"Página {page_num}: Nenhum texto extraído (pode ser um PDF escaneado)")
                    
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return ""
    
    if len(full_text.strip()) > 0:
        st.sidebar.info(f"Total de texto extraído: {len(full_text)} caracteres")
    else:
        st.sidebar.error("Nenhum texto pôde ser extraído do PDF. Verifique se o arquivo não é uma imagem escaneada.")
        
    return full_text

def parse_amount(amount_str: str) -> float:
    """
    Função CRÍTICA e mais RESTRITIVA para converter uma string de valor monetário para float.
    Agora exige a presença de uma vírgula para ser considerado um valor válido,
    evitando a conversão de números de documento.
    """
    if not isinstance(amount_str, str) or ',' not in amount_str:
        return 0.0 # Se não tiver vírgula, não é um valor monetário válido neste contexto.

    cleaned_str = amount_str.strip().replace("R$", "").strip()
    is_negative = '-' in cleaned_str or cleaned_str.endswith('-')
    cleaned_str = cleaned_str.replace('-', '')
    
    # Remove pontos de milhar e substitui a vírgula decimal por ponto
    cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
        
    try:
        value = float(cleaned_str)
        # Uma verificação de sanidade final contra valores absurdos
        if abs(value) > 100_000_000: # Ignora valores acima de 100 milhões
            return 0.0
        return -value if is_negative else value
    except (ValueError, TypeError):
        return 0.0

def parse_itau(text: str) -> List[Dict[str, Any]]:
    """Parser aprimorado para Itaú, focado em extrair o valor correto da coluna de transações."""
    transactions = []
    # Regex mais específico: Procura data, depois descrição, e captura os dois últimos valores (valor da transação e saldo)
    pattern = re.compile(r'^(\d{2}/\d{2}/\d{4})\s+(.*?)\s+(-?[\d\.]*,\d{2})\s+(-?[\d\.]*,\d{2})?$', re.MULTILINE)
    
    for match in pattern.finditer(text):
        date_str, description, val1_str, val2_str = match.groups()
        
        # O valor da transação é o primeiro capturado. val2_str é geralmente o saldo.
        amount_str = val1_str
        
        # Ignora linhas de saldo
        if any(term in description.upper() for term in ['SALDO DO DIA', 'SALDO ANTERIOR']):
            continue
            
        amount = parse_amount(amount_str)
        if amount != 0.0:
            transactions.append({
                "date": pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce'),
                "description": description.strip(),
                "amount": amount
            })
            
    return [t for t in transactions if pd.notna(t['date'])]

def parse_santander(text: str) -> List[Dict[str, Any]]:
    """Parser robusto para Santander que lida com múltiplas linhas e valida os valores."""
    transactions = []
    lines = text.split('\n')
    current_year = "2025"

    year_match = re.search(r'\b(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)/(\d{4})', text, re.IGNORECASE)
    if year_match:
        current_year = year_match.group(1)

    # Padrão para identificar uma linha que termina com um valor monetário claro
    transaction_line_pattern = re.compile(r'(-?[\d\.]*,\d{2}-?)$')
    date_pattern = re.compile(r'^\s*(\d{2}/\d{2})')

    for i, line in enumerate(lines):
        amount_match = transaction_line_pattern.search(line)
        date_match = date_pattern.search(line)
        
        # Uma linha de transação deve ter tanto uma data no início quanto um valor no final
        if date_match and amount_match:
            date_str = date_match.group(1)
            amount_str = amount_match.group(1)
            full_date_str = f"{date_str}/{current_year}"
            
            description = line.replace(date_str, "").replace(amount_str, "").strip()
            
            # Junta com linhas anteriores se elas não forem transações
            full_description_parts = [description]
            prev_idx = i - 1
            while prev_idx >= 0:
                prev_line = lines[prev_idx].strip()
                if not prev_line or "--- PAGE" in prev_line or date_pattern.search(prev_line):
                    break
                full_description_parts.insert(0, prev_line)
                prev_idx -= 1
            
            full_description = " ".join(full_description_parts).strip()

            if any(term in full_description.upper() for term in ['SALDO EM', 'SALDO ANTERIOR']):
                continue

            amount = parse_amount(amount_str)
            if amount != 0.0:
                transactions.append({
                    "date": pd.to_datetime(full_date_str, format='%d/%m/%Y', errors='coerce'),
                    "description": full_description,
                    "amount": amount
                })

    return [t for t in transactions if pd.notna(t['date'])]

def parse_inter(text: str) -> List[Dict[str, Any]]:
    """Parser aprimorado para Banco Inter com validação de valor mais estrita."""
    transactions = []
    current_date = None
    month_map = {'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04', 'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08', 'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12'}

    for line in text.split('\n'):
        date_match = re.search(r'(\d{1,2})\s+de\s+([a-zA-Zç]+)\s+de\s+(\d{4})', line, re.IGNORECASE)
        if date_match:
            day, month_name, year = date_match.groups()
            month_norm = unicodedata.normalize('NFKD', month_name.lower()).encode('ascii', 'ignore').decode('utf-8')
            month_num = month_map.get(month_norm)
            if month_num:
                current_date = pd.to_datetime(f"{day}/{month_num}/{year}", format='%d/%m/%Y', errors='coerce')
            continue

        if current_date:
            # Captura descrição, valor da transação (com R$) e saldo (com R$)
            trans_match = re.search(r'^(.*?)\s+(-?R\$\s*[\d\.]*,\d{2})\s+(-?R\$\s*[\d\.]*,\d{2})', line)
            if not trans_match:
                 # Captura descrição, valor da transação (sem R$) e saldo (com R$)
                 trans_match = re.search(r'^(.*?)\s+(-?[\d\.]*,\d{2})\s+(-?R\$\s*[\d\.]*,\d{2})', line)

            if trans_match:
                description, amount_str, _ = trans_match.groups()
                description = description.strip()

                if any(term in description.upper() for term in ['SALDO DO DIA']):
                    continue
                
                amount = parse_amount(amount_str)
                if amount != 0.0:
                    transactions.append({
                        "date": current_date,
                        "description": description,
                        "amount": amount
                    })
    return transactions

def safe_json_parse(json_str: str):
    """Tenta analisar uma string JSON, limpando-a primeiro."""
    json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
    if not json_match:
        return []
    clean_json_str = json_match.group(0)
    try:
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        st.warning("Falha no parse JSON. O Gemini pode ter retornado um formato inesperado.")
        return []

def parse_with_gemini(text: str, api_key: str) -> List[Dict[str, Any]]:
    """Usa a API do Gemini com um prompt mais específico para evitar erros."""
    if not api_key:
        st.error("A chave de API do Gemini não foi fornecida.")
        return []
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        text_sample = text[:15000] if len(text) > 15000 else text
        
        prompt = f"""
        Você é um especialista em análise de extratos bancários brasileiros. Sua tarefa é extrair transações de um texto de extrato.
        Extraia cada transação e retorne uma lista de objetos JSON com as chaves "date" (DD/MM/AAAA), "description" e "amount" (float, negativo para saídas).
        
        CRÍTICO: Ignore qualquer sequência longa de números que seja claramente um código de barras, ID de documento ou CNPJ, e não um valor monetário. Valores monetários reais quase sempre terão uma vírgula e dois dígitos para os centavos.
        
        Texto do extrato:
        ---
        {text_sample}
        ---

        Retorne APENAS a lista de objetos JSON.
        """
        
        start_time = time.time()
        response = model.generate_content(prompt)
        processing_time = time.time() - start_time
        st.sidebar.info(f"Gemini processou em {processing_time:.1f} segundos")

        transactions_json = safe_json_parse(response.text)
        
        if not transactions_json:
            st.error("Não foi possível extrair transações da resposta do Gemini.")
            st.sidebar.text_area("Resposta do Gemini (para depuração):", response.text, height=150)
            return []
            
        transactions = []
        for t in transactions_json:
            try:
                if all(k in t for k in ['date', 'description', 'amount']):
                    transactions.append({
                        "date": pd.to_datetime(t['date'], format='%d/%m/%Y', errors='coerce'),
                        "description": str(t['description']),
                        "amount": float(t['amount'])
                    })
            except (ValueError, KeyError, TypeError) as e:
                st.warning(f"Transação do Gemini ignorada devido a formato inválido: {t} | Erro: {e}")
                continue
                
        return [t for t in transactions if pd.notna(t['date'])]
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API do Gemini: {e}")
        st.error(f"Detalhes: {traceback.format_exc()}")
        return []


def detect_bank_and_parse(text: str, filename: str, gemini_key: str) -> List[Dict[str, Any]]:
    """Detecta o banco, tenta o parser por regras e usa Gemini como fallback."""
    normalized_text = unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('utf-8')
    
    parser = None
    bank_name = "Desconhecido"
    
    if 'itau' in normalized_text or 'uniclass' in normalized_text:
        bank_name = "Itaú"
        parser = parse_itau
    elif 'santander' in normalized_text:
        bank_name = "Santander"
        parser = parse_santander
    elif 'inter' in normalized_text or 'banco inter' in normalized_text:
        bank_name = "Banco Inter"
        parser = parse_inter
    
    st.sidebar.info(f"Arquivo '{filename}' identificado como: {bank_name}")
    
    transactions = []
    if parser:
        try:
            transactions = parser(text)
            st.sidebar.info(f"Parser por regras encontrou {len(transactions)} transações.")
        except Exception as e:
            st.sidebar.error(f"Erro no parser por regras: {e}")
            transactions = []

    if not transactions and gemini_key:
        st.sidebar.warning(f"Nenhuma transação via regras. Usando Gemini AI para '{filename}'...")
        with st.spinner("A IA do Gemini está analisando o extrato..."):
            gemini_transactions = parse_with_gemini(text, gemini_key)
            if gemini_transactions:
                transactions = gemini_transactions
                st.sidebar.info(f"Gemini encontrou {len(transactions)} transações.")
            else:
                st.sidebar.error("Gemini não conseguiu extrair transações.")
    
    return transactions

def categorize_transaction(description: str) -> str:
    """Categoriza uma transação com base em palavras-chave na descrição."""
    if not isinstance(description, str): return 'Outros'
    desc_lower = description.lower()
    rules = {
        'Receitas': ['pix recebido', 'sispag', 'salario', 'credito', 'deposito', 'transferencia recebida', 'ted recebida', 'remuneracao', 'desbloqueio judicial'],
        'Alimentação': ['ifood', 'restaurante', 'mercado', 'supermercado', 'lanche', 'padaria', 'acai', 'pizzaria', 'hamburguer', 'mcdonalds', 'burguer'],
        'Moradia': ['cemig', 'dmae', 'aluguel', 'condominio', 'claro', 'telefonica', 'vivo', 'tim', 'oi', 'energia', 'agua', 'internet', 'algar telecom'],
        'Transporte': ['uber', 'posto', 'gasolina', 'estacionamento', 'localiza', 'onibus', 'taxi', 'metro', '99', 'combustivel'],
        'Compras': ['lojas', 'shopping', 'mercado pag', 'havan', 'leroy', 'amazon', 'magazine', 'centauro', 'renner', 'riachuelo', 'compra cartao', 'clarisse dejesus', 'drogas', 'eletromac'],
        'Saúde': ['farmacia', 'drogaria', 'unimed', 'hospital', 'clinica', 'medico', 'dentista', 'laboratorio', 'academia', 'sua academia'],
        'Serviços & Taxas': ['pagamento fatura', 'juros', 'iof', 'seguro', 'boleto', 'crediario', 'tarifa', 'anuidade', 'taxa', 'pagamento de boleto', 'pagamento de convenio', 'prest emprestimos'],
        'Educação': ['escola', 'faculdade', 'universidade', 'curso', 'livraria', 'material escolar', 'associacao salgado'],
        'Lazer': ['cinema', 'netflix', 'spotify', 'viagem', 'hotel', 'passagem', 'streaming', 'golaco esportes'],
    }
    for category, keywords in rules.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return 'Outros'

# --- INTERFACE DA APLICAÇÃO STREAMLIT (sem alterações) ---
if 'excluded_ids' not in st.session_state:
    st.session_state.excluded_ids = set()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if "gemini_api_key" not in st.secrets:
    st.error("A chave de API do Gemini não foi encontrada nos segredos do Streamlit.")
    st.stop()

gemini_api_key = st.secrets["gemini_api_key"]

with st.sidebar:
    st.header("Controles")
    st.success("Chave de API do Gemini carregada!")
    uploaded_files = st.file_uploader("Selecione os arquivos PDF", type="pdf", accept_multiple_files=True)
    filter_term = st.text_input("Desconsiderar Titular (por nome):", help="Digite um nome para remover transações internas.")

if uploaded_files:
    current_filenames = [f.name for f in uploaded_files]
    if st.session_state.get('processed_files') != current_filenames:
        with st.spinner("Processando arquivos..."):
            all_transactions = []
            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file.getvalue())
                if text:
                    transactions = detect_bank_and_parse(text, uploaded_file.name, gemini_api_key)
                    all_transactions.extend(transactions)
                    st.sidebar.success(f"Processado {uploaded_file.name}: {len(transactions)} transações")
            if not all_transactions:
                st.error("Nenhuma transação pôde ser extraída.")
                st.stop()
            df = pd.DataFrame(all_transactions)
            df['category'] = df['description'].apply(categorize_transaction)
            df.sort_values(by='date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['id'] = df.index
            st.session_state.df_original = df
            st.session_state.processed_files = current_filenames
            st.session_state.excluded_ids = set()

    df_processed = st.session_state.df_original.copy()
    if filter_term:
        mask_filter = ~df_processed['description'].str.contains(filter_term, case=False, na=False)
        df_processed = df_processed[mask_filter]
    if st.session_state.excluded_ids:
        df_processed = df_processed[~df_processed['id'].isin(st.session_state.excluded_ids)]

    st.header("Análise Financeira")
    if not df_processed.empty:
        total_income = df_processed[df_processed['amount'] > 0]['amount'].sum()
        total_expenses = df_processed[df_processed['amount'] < 0]['amount'].sum()
        net_balance = total_income + total_expenses
        
        if 'date' in df_processed.columns and not df_processed['date'].dropna().empty:
            months_analyzed = df_processed['date'].dropna().dt.to_period('M').nunique()
        else:
            months_analyzed = 1
            
        average_income = total_income / months_analyzed if months_analyzed > 0 else 0
        presumed_capacity = average_income * 0.30

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entradas (Total)", f"R$ {total_income:,.2f}")
        col2.metric("Saídas (Total)", f"R$ {abs(total_expenses):,.2f}")
        col3.metric("Saldo Líquido", f"R$ {net_balance:,.2f}")
        col4.metric("Meses Analisados", months_analyzed)
        st.divider()
        col5, col6 = st.columns(2)
        col5.metric("Média de Entradas / Mês", f"R$ {average_income:,.2f}")
        col6.metric("Capacidade de Endividamento (30%)", f"R$ {presumed_capacity:,.2f}")
        
        st.subheader("Análise por Categoria")
        category_expenses = df_processed[df_processed['amount'] < 0].groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        category_income = df_processed[df_processed['amount'] > 0].groupby('category')['amount'].sum().sort_values(ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Despesas por Categoria**")
            if not category_expenses.empty:
                st.dataframe(category_expenses.map("R$ {:,.2f}".format), use_container_width=True)
        with col2:
            st.write("**Receitas por Categoria**")
            if not category_income.empty:
                st.dataframe(category_income.map("R$ {:,.2f}".format), use_container_width=True)

        st.subheader("Resumo Mensal")
        if 'date' in df_processed.columns:
            df_processed['Mês'] = df_processed['date'].dt.to_period('M').astype(str)
            monthly_summary = df_processed.groupby('Mês').apply(lambda x: pd.Series({'Entradas': x[x['amount'] > 0]['amount'].sum(), 'Saídas': abs(x[x['amount'] < 0]['amount'].sum()), 'Saldo': x['amount'].sum()})).reset_index()
            st.dataframe(monthly_summary.style.format({'Entradas': "R$ {:,.2f}", 'Saídas': "R$ {:,.2f}", 'Saldo': "R$ {:,.2f}"}), use_container_width=True, hide_index=True)

        st.subheader("Transações Identificadas")
        df_for_editor = df_processed.copy()
        df_for_editor['Data'] = df_for_editor['date'].dt.strftime('%d/%m/%Y')
        df_for_editor['Valor (R$)'] = df_for_editor['amount'].apply(lambda x: f"R$ {x:,.2f}")
        df_for_editor.rename(columns={'description': 'Descrição', 'category': 'Categoria'}, inplace=True)
        st.dataframe(df_for_editor[['Data', 'Descrição', 'Valor (R$)', 'Categoria']], use_container_width=True, hide_index=True, height=min(800, 35 * len(df_for_editor) + 38))
        
        with st.form("selection_form"):
            st.markdown("**Selecione transações para desconsiderar da análise:**")
            options = {f"{row['Data']} | {row['Descrição']} | {row['Valor (R$)']}": row['id'] for _, row in df_for_editor.iterrows()}
            selected_options = st.multiselect("Selecione as transações:", options=options.keys())
            if st.form_submit_button("Desconsiderar Transações Selecionadas") and selected_options:
                ids_to_exclude = {options[opt] for opt in selected_options}
                st.session_state.excluded_ids.update(ids_to_exclude)
                st.rerun()

    else:
        st.info("Nenhuma transação encontrada ou todas foram filtradas.")
else:
    st.info("Aguardando o upload de arquivos PDF para iniciar a análise.")
