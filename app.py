import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from datetime import datetime
import json
import google.generativeai as genai
import traceback

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide", page_title="Analisador de Extratos Bancários")
st.title("📊 Analisador de Extratos Bancários com IA")
st.write("Faça o upload dos seus extratos em PDF para análise completa.")

# --- FUNÇÕES PRINCIPAIS ---

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extrai texto de TODAS as páginas do PDF."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            num_pages = len(doc)
            st.sidebar.info(f"📄 PDF com {num_pages} página(s)")
            
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                full_text += page_text + "\n"
                
            st.sidebar.info(f"📊 {len(full_text)} caracteres extraídos")
                
    except Exception as e:
        st.error(f"❌ Erro ao ler PDF: {e}")
        return ""
    
    return full_text

def parse_amount(amount_str: str) -> float:
    """Converte valor monetário para float."""
    try:
        # Remove R$, pontos e espaços
        cleaned = str(amount_str).replace('R$', '').replace('.', '').replace(' ', '')
        # Substitui vírgula por ponto para decimal
        cleaned = cleaned.replace(',', '.')
        # Verifica se é negativo
        is_negative = '-' in cleaned
        cleaned = cleaned.replace('-', '')
        
        value = float(cleaned)
        return -value if is_negative else value
    except:
        return 0.0

def parse_date(date_str: str) -> datetime:
    """Tenta parsear data em múltiplos formatos."""
    try:
        # Primeiro tenta o formato mais comum DD/MM/AAAA
        try:
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            pass
        
        # Tenta outros formatos
        formats_to_try = [
            '%d-%m-%Y', '%d.%m.%Y', '%Y/%m/%d', '%d/%m/%y',
            '%d de %B de %Y', '%d de %b de %Y', '%B %Y', '%b %Y'
        ]
        
        for fmt in formats_to_try:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
                
        # Última tentativa com parser flexível
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

def extract_transactions_with_gemini(text: str, api_key: str) -> list:
    """Usa Gemini AI para extrair transações de qualquer formato de extrato."""
    if not api_key:
        st.error("❌ Chave API não configurada")
        return []
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        ANALISE ESTE EXTRATO BANCÁRIO E EXTRAIA TODAS AS TRANSAÇÕES FINANCEIRAS.

        REGRAS:
        1. Extraia CADA transação individual
        2. Data no formato DD/MM/AAAA
        3. Descrição COMPLETA da transação
        4. Valor numérico com negativo para DÉBITOS e positivo para CRÉDITOS
        5. IGNORE saldos, totais, cabeçalhos e rodapés

        FORMATO DE SAÍDA EXCLUSIVAMENTE JSON:
        [
          {{"date": "01/01/2025", "description": "DESCRIÇÃO COMPLETA", "amount": -100.00}},
          {{"date": "02/01/2025", "description": "OUTRA DESCRIÇÃO", "amount": 500.00}}
        ]

        TEXTO DO EXTRATO:
        {text[:10000]}  # Limite para não exceder tokens

        RETORNE APENAS O JSON, SEM TEXTOS ADICIONAIS.
        """

        with st.spinner("🤖 IA analisando extrato..."):
            response = model.generate_content(prompt)
            
        # Limpa a resposta
        response_text = response.text.strip()
        
        # Remove marcações de código
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        
        # Parse do JSON
        transactions = json.loads(response_text)
        
        # Processa as transações
        processed_transactions = []
        for transaction in transactions:
            try:
                date_obj = parse_date(transaction['date'])
                if pd.isna(date_obj):
                    continue
                    
                processed_transactions.append({
                    'date': date_obj,
                    'description': transaction['description'],
                    'amount': float(transaction['amount'])
                })
            except:
                continue
        
        return processed_transactions
        
    except Exception as e:
        st.error(f"❌ Erro no Gemini: {str(e)}")
        st.error("Detalhes: " + traceback.format_exc())
        return []

def categorize_transaction(description: str) -> str:
    """Categoriza transações automaticamente."""
    if not isinstance(description, str):
        return 'Outros'
    
    desc = description.lower()
    
    categories = {
        'Receita': ['salário', 'salario', 'rendimento', 'depósito', 'deposito', 'pix receb', 'transferência receb', 'ted receb'],
        'Alimentação': ['supermercado', 'mercado', 'restaurante', 'lanche', 'padaria', 'açai', 'pizza', 'ifood', 'mcdonald'],
        'Moradia': ['aluguel', 'condomínio', 'condominio', 'luz', 'água', 'agua', 'energia', 'internet', 'telefone'],
        'Transporte': ['uber', '99', 'taxi', 'ônibus', 'onibus', 'metro', 'combustível', 'combustivel', 'posto', 'estacionamento'],
        'Saúde': ['farmacia', 'drogaria', 'hospital', 'médico', 'medico', 'dentista', 'plano de saúde'],
        'Educação': ['escola', 'faculdade', 'curso', 'livro', 'material escolar'],
        'Lazer': ['cinema', 'netflix', 'spotify', 'viagem', 'hotel', 'show'],
        'Serviços': ['conserto', 'manutenção', 'manutencao', 'assistência', 'assistencia'],
        'Compras': ['shopping', 'loja', 'ecommerce', 'amazon', 'mercado livre'],
        'Investimentos': ['ação', 'acao', 'fundo', 'investimento', 'cripto', 'bitcoin'],
        'Taxas': ['tarifa', 'anuidade', 'juros', 'multa', 'iof']
    }
    
    for category, keywords in categories.items():
        if any(keyword in desc for keyword in keywords):
            return category
            
    return 'Outros'

# --- INTERFACE PRINCIPAL ---

# Configuração inicial
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'excluded_ids' not in st.session_state:
    st.session_state.excluded_ids = set()

# Verificar chave API
if "gemini_api_key" not in st.secrets:
    st.error("""
    ❌ Chave do Gemini AI não encontrada!
    
    Por favor, adicione sua chave API nas configurações do Streamlit:
    1. Acesse https://makersuite.google.com/
    2. Crie uma API key
    3. Adicione no secrets.toml: gemini_api_key = "sua-chave-aqui"
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Controles")
    st.success("✅ Gemini AI configurado")
    
    uploaded_files = st.file_uploader(
        "📤 Selecione os extratos PDF",
        type="pdf",
        accept_multiple_files=True,
        help="Podem ser vários arquivos de diferentes bancos"
    )
    
    st.markdown("---")
    filter_name = st.text_input(
        "👤 Filtrar por nome:",
        help="Digite seu nome para remover transferências internas"
    )

# Processamento principal
if uploaded_files:
    if st.button("🔄 Processar Extratos", type="primary"):
        with st.spinner("Processando arquivos..."):
            all_transactions = []
            
            for uploaded_file in uploaded_files:
                st.sidebar.info(f"📂 Processando: {uploaded_file.name}")
                
                # Extrai texto do PDF
                text = extract_text_from_pdf(uploaded_file.getvalue())
                
                if not text or len(text.strip()) < 100:
                    st.error(f"❌ Arquivo {uploaded_file.name} está vazio ou inválido")
                    continue
                
                # Extrai transações com Gemini AI
                transactions = extract_transactions_with_gemini(
                    text, 
                    st.secrets["gemini_api_key"]
                )
                
                if transactions:
                    all_transactions.extend(transactions)
                    st.sidebar.success(f"✅ {uploaded_file.name}: {len(transactions)} transações")
                else:
                    st.sidebar.error(f"❌ {uploaded_file.name}: Nenhuma transação encontrada")
            
            if all_transactions:
                # Cria DataFrame
                df = pd.DataFrame(all_transactions)
                df['category'] = df['description'].apply(categorize_transaction)
                df = df.sort_values('date', ascending=False)
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'id'}, inplace=True)
                
                st.session_state.transactions = df
                st.session_state.excluded_ids = set()
                st.success(f"✅ Análise concluída! {len(df)} transações encontradas.")
            else:
                st.error("❌ Nenhuma transação foi encontrada em nenhum arquivo.")

# Exibir resultados se existirem transações
if not st.session_state.transactions.empty:
    df = st.session_state.transactions.copy()
    
    # Aplicar filtro de nome se especificado
    if filter_name:
        mask = ~df['description'].str.contains(filter_name, case=False, na=False)
        df = df[mask]
        st.sidebar.info(f"👤 Filtrado: {filter_name}")
    
    # Remover transações excluídas
    if st.session_state.excluded_ids:
        df = df[~df['id'].isin(st.session_state.excluded_ids)]
    
    # Métricas principais
    st.header("📈 Análise Financeira")
    
    total_income = df[df['amount'] > 0]['amount'].sum()
    total_expenses = df[df['amount'] < 0]['amount'].sum()
    net_balance = total_income + total_expenses  # expenses já são negativos
    
    # Cálculo de meses - método mais robusto
    if not df.empty and 'date' in df.columns:
        df['year_month'] = df['date'].dt.to_period('M')
        unique_months = df['year_month'].nunique()
        months_analyzed = max(unique_months, 1)
        
        # Mostrar meses detectados
        months_list = sorted(df['year_month'].astype(str).unique())
        st.sidebar.info(f"📅 Meses detectados: {', '.join(months_list)}")
    else:
        months_analyzed = 1
    
    average_income = total_income / months_analyzed if months_analyzed > 0 else 0
    capacity_30 = average_income * 0.3
    
    # Layout de métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Entradas", f"R$ {total_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col2.metric("💸 Saídas", f"R$ {abs(total_expenses):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col3.metric("⚖️ Saldo", f"R$ {net_balance:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col4.metric("📅 Meses", months_analyzed)
    
    col5, col6 = st.columns(2)
    col5.metric("📊 Média Mensal", f"R$ {average_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    col6.metric("🎯 Capacidade 30%", f"R$ {capacity_30:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    
    # Análise por categoria
    st.subheader("🗂️ Análise por Categoria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📉 Despesas por Categoria**")
        expenses = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        for category, amount in expenses.items():
            st.write(f"{category}: R$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    
    with col2:
        st.write("**📈 Receitas por Categoria**")
        income = df[df['amount'] > 0].groupby('category')['amount'].sum().sort_values(ascending=False)
        for category, amount in income.items():
            st.write(f"{category}: R$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    
    # Resumo mensal
    st.subheader("📅 Resumo Mensal")
    
    if not df.empty and 'year_month' in df.columns:
        monthly_summary = df.groupby('year_month').agg({
            'amount': [
                ('Entradas', lambda x: x[x > 0].sum()),
                ('Saídas', lambda x: x[x < 0].sum()),
                ('Saldo', 'sum')
            ]
        }).round(2)
        
        monthly_summary.columns = ['Entradas', 'Saídas', 'Saldo']
        monthly_summary['Mês'] = monthly_summary.index.astype(str)
        monthly_summary = monthly_summary[['Mês', 'Entradas', 'Saídas', 'Saldo']]
        
        # Formatar valores
        for col in ['Entradas', 'Saídas', 'Saldo']:
            monthly_summary[col] = monthly_summary[col].apply(
                lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            )
        
        st.dataframe(monthly_summary, use_container_width=True, hide_index=True)
    
    # Todas as transações
    st.subheader("💳 Todas as Transações")
    
    df_display = df.copy()
    df_display['Data'] = df_display['date'].dt.strftime('%d/%m/%Y')
    df_display['Valor'] = df_display['amount'].apply(
        lambda x: f"R$ {abs(x):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') + 
                 (' 🔴' if x < 0 else ' 🟢')
    )
    
    st.dataframe(
        df_display[['Data', 'description', 'Valor', 'category']].rename(
            columns={'description': 'Descrição', 'category': 'Categoria'}
        ),
        use_container_width=True,
        hide_index=True,
        height=min(600, 35 * len(df_display) + 38)
    )
    
    # Botão para exportar dados
    if st.button("📤 Exportar para Excel"):
        csv = df_display[['Data', 'Descrição', 'Valor', 'Categoria']].to_csv(index=False)
        st.download_button(
            label="⬇️ Baixar CSV",
            data=csv,
            file_name="extrato_analisado.csv",
            mime="text/csv"
        )

else:
    st.info("📁 Faça o upload dos extratos PDF e clique em 'Processar Extratos'")

# Mensagem de status
if uploaded_files and st.session_state.transactions.empty:
    st.warning("""
    ⚠️ Nenhuma transação foi encontrada. Isso pode acontecer por:
    
    1. 🔑 Problema com a chave API do Gemini
    2. 📄 PDFs escaneados (imagens) em vez de texto
    3. 🏦 Formato de extrato muito diferente
    4. 🌐 Problema de conexão com a API
    
    **Solução:** Verifique se os PDFs contêm texto selecionável e tente novamente.
    """)
