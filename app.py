import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict, Any
import unicodedata
import json
import google.generativeai as genai

# --- FUNÇÕES DE LÓGICA DE ANÁLISE ---

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extrai texto de um arquivo PDF, preservando as quebras de linha."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
    return full_text

def parse_amount(amount_str: str) -> float:
    """Converte uma string de valor monetário para float."""
    if not isinstance(amount_str, str):
        return 0.0
    cleaned_str = str(amount_str).replace('R$', '').strip()
    cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
    try:
        return float(cleaned_str)
    except (ValueError, TypeError):
        return 0.0

def parse_itau(text: str) -> List[Dict[str, Any]]:
    """Parser robusto para extratos do Itaú que analisa linha por linha."""
    transactions = []
    date_regex = re.compile(r'^(\d{2}\/\d{2}\/\d{4})')
    amount_regex = re.compile(r'(-?[\d\.]*,\d{2})$')

    for line in text.split('\n'):
        line = line.strip()
        date_match = date_regex.search(line)
        amount_match = amount_regex.search(line)

        if date_match and amount_match:
            date_str = date_match.group(1)
            amount_str = amount_match.group(1)
            start_index = date_match.end()
            end_index = amount_match.start()
            description = line[start_index:end_index].strip()

            if description.upper() in ['SALDO DO DIA', 'SALDO ANTERIOR', 'LANÇAMENTOS'] or not description:
                continue
            
            transactions.append({
                "date": pd.to_datetime(date_str, format='%d/%m/%Y'),
                "description": description,
                "amount": parse_amount(amount_str)
            })
    return transactions

def parse_inter(text: str) -> List[Dict[str, Any]]:
    """Parser para extratos do Banco Inter."""
    transactions = []
    current_date = None
    date_header_regex = re.compile(r'(\d{1,2} de [A-Za-zç]+ de \d{4})')
    for line in text.split('\n'):
        line = line.strip()
        date_match = date_header_regex.search(line)
        if date_match:
            date_str = date_match.group(1)
            try:
                month_map = {'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 'abril': 'April', 'maio': 'May', 'junho': 'June', 'julho': 'July', 'agosto': 'August', 'setembro': 'September', 'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'}
                for pt, en in month_map.items():
                    date_str = date_str.replace(pt, en)
                current_date = datetime.strptime(date_str, '%d de %B de %Y')
            except ValueError:
                continue
        elif current_date:
            parts = line.split()
            if len(parts) > 1 and 'R$' in parts[-1]:
                amount_str = parts[-1]
                description = " ".join(parts[:-1])
                if "Saldo por transação" in description:
                    continue
                transactions.append({"date": current_date, "description": description.strip(), "amount": parse_amount(amount_str)})
    return transactions

def parse_with_gemini(text: str, api_key: str) -> List[Dict[str, Any]]:
    """Usa a API do Gemini para extrair transações de um texto de extrato."""
    if not api_key:
        st.error("A chave de API do Gemini não foi fornecida.")
        return []
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Você é um especialista em análise de dados financeiros. Sua tarefa é extrair transações de um texto de extrato bancário.
        O texto a seguir é o conteúdo de um extrato em PDF. Identifique cada transação e retorne uma lista de objetos JSON.
        Cada objeto deve ter EXATAMENTE as seguintes chaves: "date" (no formato "DD/MM/AAAA"), "description" (a descrição completa) e "amount" (o valor como um número, usando ponto como separador decimal, e negativo para saídas).
        Ignore linhas de saldo, cabeçalhos ou qualquer outra informação que não seja uma transação.

        Texto do extrato:
        ---
        {text}
        ---

        Retorne APENAS a lista de objetos JSON. Exemplo de saída:
        [
          {{"date": "30/06/2025", "description": "PIX TRANSF BRUNO C28/06", "amount": -1500.00}},
          {{"date": "30/06/2025", "description": "SISPAG PIX H2 ESTACIONAMENTO...", "amount": 1500.00}}
        ]
        """
        
        response = model.generate_content(prompt)
        # Limpa a resposta para garantir que seja um JSON válido
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        
        transactions_json = json.loads(cleaned_response)
        
        # Converte os dados para o formato correto
        transactions = []
        for t in transactions_json:
            transactions.append({
                "date": pd.to_datetime(t['date'], format='%d/%m/%Y'),
                "description": t['description'],
                "amount": float(t['amount'])
            })
        return transactions
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API do Gemini: {e}")
        return []


def detect_bank_and_parse(text: str, filename: str, gemini_key: str) -> List[Dict[str, Any]]:
    """Detecta o banco, tenta o parser normal e usa Gemini como fallback."""
    nfkd_form = unicodedata.normalize('NFKD', text.lower())
    normalized_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    parser = None
    if 'itau uniclass' in normalized_text or 'itau' in normalized_text:
        st.sidebar.info(f"Arquivo '{filename}' identificado como: Itaú")
        parser = parse_itau
    elif 'banco inter' in normalized_text:
        st.sidebar.info(f"Arquivo '{filename}' identificado como: Banco Inter")
        parser = parse_inter
    
    transactions = []
    if parser:
        transactions = parser(text)

    # Se o parser padrão falhou ou extraiu poucas transações, usa Gemini
    if len(transactions) < 5 and gemini_key:
        st.sidebar.warning("Parser padrão falhou. Usando análise com Gemini AI...")
        with st.spinner("A IA do Gemini está analisando o extrato..."):
            transactions = parse_with_gemini(text, gemini_key)
    elif not parser:
        st.sidebar.warning(f"Banco não reconhecido para '{filename}'. Tentando com Gemini AI...")
        with st.spinner("A IA do Gemini está analisando o extrato..."):
            transactions = parse_with_gemini(text, gemini_key)

    return transactions

def categorize_transaction(description: str) -> str:
    """Categoriza uma transação com base em palavras-chave na descrição."""
    desc_lower = description.lower()
    rules = {
        'Receitas': ['pix recebido', 'sispag', 'salário', 'credito'],
        'Alimentação': ['ifood', 'restaurante', 'mercado', 'supermercado', 'lanche'],
        'Moradia': ['cemig', 'dmae', 'aluguel', 'condominio', 'claro', 'telefonica'],
        'Transporte': ['uber', 'posto', 'gasolina', 'estacionamento', 'localiza'],
        'Compras': ['lojas', 'shopping', 'mercado pag', 'havan', 'leroy'],
        'Saúde': ['farmacia', 'drogaria', 'unimed', 'hospital'],
        'Serviços & Taxas': ['pagamento fatura', 'juros', 'iof', 'seguro', 'boleto', 'crediario', 'int uniclass vs', 'juros limite da conta'],
    }
    for category, keywords in rules.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return 'Outros'

# --- INTERFACE DA APLICAÇÃO STREAMLIT ---

st.set_page_config(layout="wide", page_title="Analisador de Extratos Bancários")

st.title("📊 Analisador de Extratos Bancários com IA")
st.write("Faça o upload dos seus extratos em PDF. A análise será feita por regras e, se necessário, pela IA do Gemini.")

# Inicializa o estado da sessão
if 'excluded_ids' not in st.session_state:
    st.session_state.excluded_ids = set()

# --- Sidebar para Controles ---
with st.sidebar:
    st.header("Controles")
    gemini_api_key = st.text_input("Chave de API do Google Gemini", type="password", help="Sua chave é necessária para a análise com IA.")
    
    uploaded_files = st.file_uploader(
        "Selecione os arquivos PDF",
        type="pdf",
        accept_multiple_files=True
    )
    
    filter_term = st.text_input(
        "Desconsiderar Titular (por nome):",
        help="Digite um nome para remover transações internas da análise."
    )

# --- Lógica Principal da Aplicação ---
if uploaded_files:
    current_filenames = [f.name for f in uploaded_files]
    
    if 'df_original' not in st.session_state or st.session_state.get('processed_files') != current_filenames:
        with st.spinner("Processando arquivos..."):
            all_transactions = []
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                text = extract_text_from_pdf(file_content)
                transactions = detect_bank_and_parse(text, uploaded_file.name, gemini_api_key)
                all_transactions.extend(transactions)

            if not all_transactions:
                st.error("Nenhuma transação pôde ser extraída. Verifique os PDFs ou sua chave de API do Gemini.")
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
        mask_filter = ~df_processed['description'].str.contains(filter_term, case=False, regex=False)
        removed_by_name = df_processed[~mask_filter]
        df_processed = df_processed[mask_filter]
        st.sidebar.info(f"{len(removed_by_name)} transações removidas pelo filtro de nome.")

    if st.session_state.excluded_ids:
        df_processed = df_processed[~df_processed['id'].isin(st.session_state.excluded_ids)]

    # --- Exibição dos Resultados ---
    st.header("Análise Financeira")

    total_income = df_processed[df_processed['amount'] > 0]['amount'].sum()
    total_expenses = df_processed[df_processed['amount'] < 0]['amount'].sum()
    months_analyzed = df_processed['date'].dt.to_period('M').nunique() if not df_processed.empty else 0
    average_income = total_income / months_analyzed if months_analyzed > 0 else 0
    presumed_income = average_income * 0.30

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entradas (Total)", f"R$ {total_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col2.metric("Saídas (Total)", f"R$ {abs(total_expenses):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col3.metric("Ticket Médio / Mês", f"R$ {average_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col4.metric("Capacidade 30%", f"R$ {presumed_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

    st.markdown("---")
    
    st.subheader("Resumo Mensal")
    if not df_processed.empty:
        df_processed['Mês'] = df_processed['date'].dt.strftime('%Y-%m')
        monthly_summary = df_processed.groupby('Mês').apply(lambda x: pd.Series({
            'Entradas': x[x['amount'] > 0]['amount'].sum(),
            'Saídas': x[x['amount'] < 0]['amount'].sum(),
            'Saldo': x['amount'].sum()
        })).reset_index()
        
        for col_name in ['Entradas', 'Saídas', 'Saldo']:
            monthly_summary[col_name] = monthly_summary[col_name].map("R$ {:,.2f}".format)
        st.dataframe(monthly_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Não há dados para exibir o resumo mensal.")

    st.markdown("---")

    st.subheader("Transações Identificadas")
    
    df_for_editor = df_processed.copy()
    df_for_editor['Data'] = df_for_editor['date'].dt.strftime('%d/%m/%Y')
    df_for_editor['Valor (R$)'] = df_for_editor['amount'].map("{:,.2f}".format)
    df_for_editor.rename(columns={'description': 'Descrição', 'category': 'Categoria'}, inplace=True)
    
    # Usando o novo st.data_editor para seleção
    selection = st.data_editor(
        df_for_editor[['Data', 'Descrição', 'Valor (R$)', 'Categoria']],
        key="data_editor",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic"
    )
    
    # Botão para desconsiderar transações selecionadas
    if st.button("Desconsiderar Transação(ões) Selecionada(s)"):
        # A seleção é acessada de forma diferente no novo data_editor
        # Infelizmente, o data_editor não tem um callback direto para seleção de linhas.
        # A abordagem mais simples é usar um widget separado para obter os índices.
        st.warning("A seleção de linhas para exclusão direta no `st.data_editor` ainda não é suportada de forma ideal. Use o filtro por nome na barra lateral por enquanto.")


else:
    st.info("Aguardando o upload de arquivos PDF para iniciar a análise.")

