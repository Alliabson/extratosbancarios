import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
from enum import Enum
import numpy as np

# --- CONSTANTES E CONFIGURAÇÕES ---
class BankType(Enum):
    ITAU = "Itaú"
    BRADESCO = "Bradesco"
    SANTANDER = "Santander"
    INTER = "Banco Inter"
    CAIXA = "Caixa Econômica"
    BB = "Banco do Brasil"
    UNKNOWN = "Desconhecido"

# Mapeamento de meses em português para inglês
MONTH_MAPPING = {
    'janeiro': 'January',
    'fevereiro': 'February',
    'março': 'March',
    'abril': 'April',
    'maio': 'May',
    'junho': 'June',
    'julho': 'July',
    'agosto': 'August',
    'setembro': 'September',
    'outubro': 'October',
    'novembro': 'November',
    'dezembro': 'December'
}

# --- FUNÇÕES AUXILIARES ---
def normalize_text(text: str) -> str:
    """Normaliza texto removendo acentos e caracteres especiais."""
    nfkd_form = unicodedata.normalize('NFKD', text.lower())
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def parse_amount(amount_str: str) -> float:
    """Converte uma string de valor monetário para float de forma robusta."""
    if not isinstance(amount_str, str):
        return 0.0
    
    # Remove caracteres não numéricos exceto -,. e substitui vírgula por ponto
    cleaned = re.sub(r'[^\d,-]', '', amount_str.strip())
    cleaned = cleaned.replace('.', '').replace(',', '.')
    
    # Verifica se é negativo (com - ou entre parênteses)
    is_negative = '-' in cleaned or '(' in cleaned
    
    try:
        value = abs(float(re.sub(r'[^\d.]', '', cleaned)))
        return -value if is_negative else value
    except (ValueError, TypeError):
        return 0.0

def detect_bank(text: str) -> BankType:
    """Identifica o banco com base em padrões no texto."""
    normalized = normalize_text(text)
    
    bank_patterns = {
        BankType.ITAU: r'itau|itau uniclass|itau pessoal',
        BankType.BRADESCO: r'bradesco|banco bradesco',
        BankType.SANTANDER: r'santander|banco santander',
        BankType.INTER: r'banco inter|inter medium',
        BankType.CAIXA: r'caixa economica|caixa federal',
        BankType.BB: r'banco do brasil|bb|banco brasil'
    }
    
    for bank, pattern in bank_patterns.items():
        if re.search(pattern, normalized):
            return bank
    
    return BankType.UNKNOWN

# --- PARSERS PARA DIFERENTES BANCOS ---
def parse_itau(text: str) -> List[Dict[str, Any]]:
    """Parser robusto para extratos do Itaú."""
    transactions = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Padrão para linha de transação: data, descrição e valor
    pattern = re.compile(
        r'^(\d{2}/\d{2}/\d{4})\s+(.*?)\s+(-?[\d.,]+)\s*$'
    )
    
    for line in lines:
        match = pattern.search(line)
        if match:
            date_str, description, amount_str = match.groups()
            
            # Ignorar linhas de saldo
            if "saldo do dia" in description.lower():
                continue
                
            try:
                date = datetime.strptime(date_str, '%d/%m/%Y')
                amount = parse_amount(amount_str)
                
                transactions.append({
                    'date': date,
                    'description': description.strip(),
                    'amount': amount,
                    'bank': 'Itaú'
                })
            except ValueError:
                continue
                
    return transactions

def parse_inter(text: str) -> List[Dict[str, Any]]:
    """Parser para extratos do Banco Inter."""
    transactions = []
    current_date = None
    
    # Padrão para cabeçalho de data (ex: "10 de Fevereiro de 2025")
    date_pattern = re.compile(r'(\d{1,2})\s+de\s+([a-zA-Zç]+)\s+de\s+(\d{4})')
    
    # Padrão para linha de transação (descrição seguida de valor R$)
    transaction_pattern = re.compile(r'^(.*?)\s+(-?R\$\s*[\d.,]+)\s*$')
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Verifica se é um cabeçalho de data
        date_match = date_pattern.search(line)
        if date_match:
            day, month_pt, year = date_match.groups()
            month_en = MONTH_MAPPING.get(month_pt.lower(), month_pt)
            try:
                current_date = datetime.strptime(
                    f"{day} {month_en} {year}", 
                    '%d %B %Y'
                )
            except ValueError:
                continue
            continue
            
        # Se temos uma data atual, procura por transações
        if current_date:
            trans_match = transaction_pattern.search(line)
            if trans_match:
                description, amount_str = trans_match.groups()
                
                # Ignorar linhas de saldo
                if "saldo do dia" in description.lower():
                    continue
                    
                transactions.append({
                    'date': current_date,
                    'description': description.strip(),
                    'amount': parse_amount(amount_str),
                    'bank': 'Banco Inter'
                })
    
    return transactions

def parse_generic(text: str) -> List[Dict[str, Any]]:
    """Parser genérico que tenta identificar transações em qualquer formato."""
    transactions = []
    
    # Padrão genérico para data no início da linha
    date_pattern = re.compile(
        r'^(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|'  # DD/MM/YYYY ou similar
        r'(\d{1,2}\s+de\s+[a-zA-Zç]+\s+de\s+\d{4})'  # "10 de Fevereiro de 2025"
    )
    
    # Padrão para valores monetários (R$, $, ou números com vírgula/ponto)
    amount_pattern = re.compile(
        r'(-?\s*(?:R\$\s*)?[\d.,]+\s*(?:R\$\s*)?)'  # R$ 1.234,56 ou 1,234.56
    )
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        # Tenta encontrar uma data
        date_match = date_pattern.search(line)
        if not date_match:
            continue
            
        # Tenta encontrar um valor monetário
        amount_match = amount_pattern.search(line)
        if not amount_match:
            continue
            
        # A descrição é o que está entre a data e o valor
        date_str = date_match.group()
        amount_str = amount_match.group()
        
        start_idx = date_match.end()
        end_idx = amount_match.start()
        description = line[start_idx:end_idx].strip()
        
        # Tenta parsear a data
        try:
            # Formato DD/MM/YYYY
            if '/' in date_str:
                date = datetime.strptime(date_str, '%d/%m/%Y')
            # Formato "10 de Fevereiro de 2025"
            elif 'de' in date_str:
                day, month_pt, _, year = date_str.split()
                month_en = MONTH_MAPPING.get(month_pt.lower(), month_pt)
                date = datetime.strptime(
                    f"{day} {month_en} {year}", 
                    '%d %B %Y'
                )
            else:
                continue
        except ValueError:
            continue
            
        # Parseia o valor
        amount = parse_amount(amount_str)
        
        transactions.append({
            'date': date,
            'description': description,
            'amount': amount,
            'bank': 'Desconhecido'
        })
    
    return transactions

# --- CATEGORIZAÇÃO DE TRANSAÇÕES ---
def categorize_transaction(description: str) -> str:
    """Categoriza transações com base em palavras-chave."""
    desc_lower = normalize_text(description)
    
    categories = {
        'Salário': ['salario', 'salário', 'pagamento', 'pro labore', 'renda'],
        'Transferência': ['pix', 'transferencia', 'transf', 'ted', 'doc'],
        'Alimentação': ['supermercado', 'mercado', 'ifood', 'restaurante', 'lanche', 'padaria', 'açougue', 'hortifruti'],
        'Moradia': ['aluguel', 'condominio', 'agua', 'luz', 'energia', 'internet', 'telefone', 'net', 'vivo', 'claro', 'oi'],
        'Transporte': ['uber', 'taxi', 'onibus', 'metro', 'combustivel', 'gasolina', 'posto', 'estacionamento', 'auto peças'],
        'Saúde': ['farmacia', 'drogaria', 'hospital', 'clinica', 'medico', 'dentista', 'plano de saude', 'unimed'],
        'Lazer': ['cinema', 'netflix', 'spotify', 'viagem', 'hotel', 'passagem', 'parque', 'shows'],
        'Educação': ['escola', 'curso', 'faculdade', 'universidade', 'material escolar', 'livraria'],
        'Compras': ['loja', 'shopping', 'ecommerce', 'amazon', 'magazine', 'vestuario', 'roupa', 'calcado'],
        'Investimentos': ['aplicacao', 'investimento', 'tesouro direto', 'acoes', 'fundo', 'cdb', 'lci', 'lca'],
        'Serviços Financeiros': ['tarifa', 'juros', 'multa', 'boleto', 'financiamento', 'emprestimo', 'seguro', 'consorcio'],
        'Outros': []
    }
    
    for category, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
            
    # Verifica padrões específicos
    if re.search(r'imposto|taxa|contribuicao|iptu|ipva|irpf', desc_lower):
        return 'Impostos e Taxas'
    if re.search(r'cartao|fatura|credito|debito', desc_lower):
        return 'Cartão de Crédito'
    if re.search(r'doacao|contribuicao|oferta', desc_lower):
        return 'Doações'
    
    return 'Outros'

# --- ANÁLISE FINANCEIRA ---
def financial_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Realiza análise financeira completa do DataFrame de transações."""
    if df.empty:
        return {}
    
    analysis = {}
    
    # Período analisado
    min_date = df['date'].min()
    max_date = df['date'].max()
    analysis['period'] = f"{min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}"
    
    # Totais
    analysis['total_income'] = df[df['amount'] > 0]['amount'].sum()
    analysis['total_expenses'] = df[df['amount'] < 0]['amount'].sum()
    analysis['net_balance'] = analysis['total_income'] + analysis['total_expenses']  # soma porque expenses é negativo
    
    # Por mês
    df['month_year'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month_year')['amount'].agg(['sum', 'count'])
    analysis['avg_monthly_income'] = df[df['amount'] > 0].groupby('month_year')['amount'].sum().mean()
    analysis['avg_monthly_expense'] = df[df['amount'] < 0].groupby('month_year')['amount'].sum().mean()
    
    # Por categoria
    df['category'] = df['description'].apply(categorize_transaction)
    analysis['category_expenses'] = df[df['amount'] < 0].groupby('category')['amount'].sum().sort_values()
    analysis['category_income'] = df[df['amount'] > 0].groupby('category')['amount'].sum().sort_values(ascending=False)
    
    # Top transações
    analysis['top_incomes'] = df[df['amount'] > 0].nlargest(5, 'amount')
    analysis['top_expenses'] = df[df['amount'] < 0].nsmallest(5, 'amount')
    
    return analysis

# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(
        layout="wide", 
        page_title="Analisador de Extratos Bancários",
        page_icon="📊"
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .stMetric {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 10px;
        background-color: #f8f9fa;
    }
    .stMetric label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stMetric div {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    .stDataFrame {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Analisador de Extratos Bancários")
    st.markdown("Faça o upload dos seus extratos em formato PDF para uma análise financeira detalhada.")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("Controles")
        
        uploaded_files = st.file_uploader(
            "Selecione os arquivos PDF",
            type="pdf",
            accept_multiple_files=True,
            help="Você pode selecionar múltiplos arquivos de uma vez."
        )
        
        filter_name = st.text_input(
            "Filtrar transações por nome:",
            help="Digite um nome para remover transações internas (ex: transferências entre contas)."
        )
        
        st.markdown("---")
        st.markdown("**Configurações de análise**")
        show_raw_data = st.checkbox("Mostrar dados brutos", False)
        show_category_analysis = st.checkbox("Análise por categoria", True)
    
    # --- Processamento dos arquivos ---
    if uploaded_files:
        with st.spinner("Processando arquivos... Por favor aguarde."):
            all_transactions = []
            
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                text = extract_text_from_pdf(file_content)
                
                bank = detect_bank(text)
                st.info(f"Arquivo '{uploaded_file.name}' identificado como: {bank.value}")
                
                if bank == BankType.ITAU:
                    transactions = parse_itau(text)
                elif bank == BankType.INTER:
                    transactions = parse_inter(text)
                else:
                    transactions = parse_generic(text)
                    st.warning(f"Usando parser genérico para o arquivo '{uploaded_file.name}'. "
                              "Algumas transações podem não ser identificadas corretamente.")
                
                all_transactions.extend(transactions)
            
            if not all_transactions:
                st.error("Nenhuma transação foi identificada nos arquivos. Verifique se são extratos bancários válidos.")
                st.stop()
            
            df = pd.DataFrame(all_transactions)
            
            # Aplicar filtro se fornecido
            if filter_name:
                mask = ~df['description'].str.contains(filter_name, case=False, regex=False)
                removed_count = len(df) - mask.sum()
                df = df[mask]
                if removed_count > 0:
                    st.sidebar.success(f"Filtradas {removed_count} transações contendo '{filter_name}'.")
            
            # Categorizar transações
            df['category'] = df['description'].apply(categorize_transaction)
            
            # Análise financeira
            analysis = financial_analysis(df)
            
            # Armazenar no estado da sessão
            st.session_state.df = df
            st.session_state.analysis = analysis
    
    # --- Exibição dos resultados ---
    if 'df' in st.session_state and 'analysis' in st.session_state:
        df = st.session_state.df
        analysis = st.session_state.analysis
        
        st.header("📈 Visão Geral Financeira")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Receitas", 
                   f"R$ {analysis['total_income']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col2.metric("Total de Despesas", 
                   f"R$ {abs(analysis['total_expenses']):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col3.metric("Saldo Líquido", 
                   f"R$ {analysis['net_balance']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
                   delta_color="inverse")
        col4.metric("Período Analisado", analysis['period'])
        
        st.markdown("---")
        
        # Gráficos e análises
        if show_category_analysis:
            st.subheader("📊 Análise por Categoria")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Despesas por Categoria**")
                if not analysis['category_expenses'].empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    analysis['category_expenses'].plot(kind='barh', ax=ax, color='#ff6b6b')
                    ax.set_xlabel('Valor (R$)')
                    ax.set_ylabel('')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                else:
                    st.info("Nenhuma despesa categorizada encontrada.")
            
            with col2:
                st.markdown("**Receitas por Categoria**")
                if not analysis['category_income'].empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    analysis['category_income'].plot(kind='barh', ax=ax, color='#51cf66')
                    ax.set_xlabel('Valor (R$)')
                    ax.set_ylabel('')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                else:
                    st.info("Nenhuma receita categorizada encontrada.")
            
            st.markdown("---")
        
        # Transações recentes
        st.subheader("💸 Últimas Transações")
        st.dataframe(
            df.sort_values('date', ascending=False).head(20)[['date', 'description', 'amount', 'category']].rename(
                columns={
                    'date': 'Data', 
                    'description': 'Descrição', 
                    'amount': 'Valor (R$)', 
                    'category': 'Categoria'
                }
            ).style.format({'Valor (R$)': "{:,.2f}"}),
            height=500,
            use_container_width=True
        )
        
        # Dados brutos se solicitado
        if show_raw_data:
            st.subheader("📝 Dados Brutos")
            st.dataframe(df)
    
    else:
        st.info("⏳ Aguardando upload de arquivos PDF para análise...")
        st.image("https://via.placeholder.com/800x400?text=Fa%C3%A7a+upload+de+extratos+banc%C3%A1rios+em+PDF", 
                use_column_width=True)

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extrai texto de um arquivo PDF."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
    return full_text

if __name__ == "__main__":
    main()
