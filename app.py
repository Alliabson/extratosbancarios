import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import io
import unicodedata

# --- FUNÇÕES DE LÓGICA DE ANÁLISE (Otimizadas para Streamlit) ---

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extrai texto de um arquivo PDF a partir de seu conteúdo em bytes."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                # Usar get_text() sem argumentos preserva melhor as quebras de linha.
                full_text += page.get_text()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
    return full_text

def parse_amount(amount_str: str) -> float:
    """Converte uma string de valor monetário para float."""
    if not isinstance(amount_str, str):
        return 0.0
    # Remove 'R$', espaços, e troca o ponto de milhar por nada, e a vírgula decimal por ponto.
    cleaned_str = amount_str.replace('R$', '').strip()
    cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
    try:
        return float(cleaned_str)
    except (ValueError, TypeError):
        return 0.0

def parse_itau(text: str) -> List[Dict[str, Any]]:
    """Analisa o texto de um extrato do Itaú de forma mais robusta."""
    transactions = []
    # Regex para encontrar a data no início da linha
    date_regex = re.compile(r'^(\d{2}\/\d{2}\/\d{4})')
    # Regex para encontrar o valor no final da linha
    amount_regex = re.compile(r'(-?[\d\.]*,\d{2})$')

    for line in text.split('\n'):
        line = line.strip()
        
        date_match = date_regex.search(line)
        amount_match = amount_regex.search(line)

        # A linha é considerada uma transação se tiver uma data no início e um valor no final
        if date_match and amount_match:
            date_str = date_match.group(1)
            amount_str = amount_match.group(1)
            
            # O que está entre a data e o valor é a descrição
            start_index = date_match.end()
            end_index = amount_match.start()
            
            description = line[start_index:end_index].strip()

            # Ignora linhas que são apenas informativas ou sem descrição
            if description.upper() in ['SALDO DO DIA', 'SALDO ANTERIOR', 'LANÇAMENTOS'] or not description:
                continue
            
            transactions.append({
                "date": pd.to_datetime(date_str, format='%d/%m/%Y'),
                "description": description,
                "amount": parse_amount(amount_str)
            })
                
    return transactions

def parse_inter(text: str) -> List[Dict[str, Any]]:
    """Analisa o texto de um extrato do Inter e extrai as transações."""
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

def detect_bank_and_parse(text: str, filename: str) -> List[Dict[str, Any]]:
    """Detecta o banco e chama a função de parsing apropriada."""
    nfkd_form = unicodedata.normalize('NFKD', text.lower())
    normalized_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    if 'itau uniclass' in normalized_text or 'itau' in normalized_text:
        st.info(f"Arquivo '{filename}' identificado como: Itaú")
        return parse_itau(text)
    elif 'banco inter' in normalized_text:
        st.info(f"Arquivo '{filename}' identificado como: Banco Inter")
        return parse_inter(text)
    else:
        st.warning(f"Banco não reconhecido para o arquivo '{filename}'.")
        return []

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

st.title("📊 Analisador de Extratos Bancários")
st.write("Faça o upload dos seus extratos em formato PDF para uma análise financeira detalhada.")

# --- Sidebar para Controles ---
with st.sidebar:
    st.header("Controles")
    uploaded_files = st.file_uploader(
        "Selecione os arquivos PDF",
        type="pdf",
        accept_multiple_files=True
    )
    
    filter_term = st.text_input(
        "Filtrar e remover transações por nome:",
        help="Digite um nome (ex: Herbert) para remover transações internas da análise."
    )

# --- Lógica Principal da Aplicação ---
if uploaded_files:
    current_filenames = [f.name for f in uploaded_files]
    
    if 'df_original' not in st.session_state or st.session_state.get('processed_files') != current_filenames:
        with st.spinner("Processando arquivos... Isso pode levar alguns segundos."):
            all_transactions = []
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                text = extract_text_from_pdf(file_content)
                transactions = detect_bank_and_parse(text, uploaded_file.name)
                all_transactions.extend(transactions)

            if not all_transactions:
                st.error("Nenhuma transação pôde ser extraída dos arquivos. Verifique se os PDFs são extratos bancários válidos.")
                st.stop()

            df = pd.DataFrame(all_transactions)
            df['category'] = df['description'].apply(categorize_transaction)
            st.session_state.df_original = df
            st.session_state.processed_files = current_filenames

    df_display = st.session_state.df_original
    if filter_term:
        df_display = df_display[~df_display['description'].str.contains(filter_term, case=False, regex=False)]
        removed_count = len(st.session_state.df_original) - len(df_display)
        st.sidebar.info(f"{removed_count} transações contendo '{filter_term}' foram removidas da análise.")

    # --- Exibição dos Resultados ---
    st.header("Análise Financeira")

    total_income = df_display[df_display['amount'] > 0]['amount'].sum()
    total_expenses = df_display[df_display['amount'] < 0]['amount'].sum()
    months_analyzed = df_display['date'].dt.to_period('M').nunique() if not df_display.empty else 0
    average_income = total_income / months_analyzed if months_analyzed > 0 else 0
    presumed_income = average_income * 0.30

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Receitas", f"R$ {total_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col2.metric("Total de Despesas", f"R$ {total_expenses:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col3.metric("Meses Analisados", f"{months_analyzed}")
    col4.metric("Renda Média Mensal", f"R$ {average_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col5.metric("Renda Presumida (30%)", f"R$ {presumed_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

    st.markdown("---")

    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        st.subheader("Despesas por Categoria")
        expenses_df = df_display[df_display['amount'] < 0].copy()
        if not expenses_df.empty:
            expenses_df['amount'] = expenses_df['amount'].abs()
            category_expenses = expenses_df.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(x=category_expenses.values, y=category_expenses.index, ax=ax, palette="viridis")
            ax.set_xlabel('Valor (R$)')
            ax.set_ylabel('')
            st.pyplot(fig)
        else:
            st.info("Não há dados de despesas para exibir no gráfico.")

    with col_table:
        st.subheader("Todas as Transações")
        df_display_formatted = df_display.copy()
        df_display_formatted['amount'] = df_display_formatted['amount'].apply(
            lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        )
        st.dataframe(df_display_formatted[['date', 'description', 'amount', 'category']].rename(
            columns={'date': 'Data', 'description': 'Descrição', 'amount': 'Valor (R$)', 'category': 'Categoria'}
        ))

else:
    st.info("Aguardando o upload de arquivos PDF para iniciar a análise.")

