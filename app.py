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
st.write("Faça o upload dos seus extratos em PDF. A análise será feita pela IA do Gemini para garantir a máxima compatibilidade.")

# --- FUNÇÕES DE LÓGICA DE ANÁLISE ---

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extrai texto de TODAS as páginas de um arquivo PDF."""
    full_text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text() + "\n"
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return ""
    
    if not full_text.strip():
        st.sidebar.warning("Nenhum texto extraído. O PDF é provavelmente uma imagem escaneada. A IA tentará fazer a leitura óptica (OCR).")
        
    return full_text

def safe_json_parse(json_str: str) -> List[Dict[str, Any]]:
    """Tenta analisar uma string JSON que pode estar mal formatada, limpando-a primeiro."""
    json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
    if not json_match:
        st.warning("A resposta da IA não continha um formato de lista JSON válido.")
        return []
    
    clean_json_str = json_match.group(0)
    try:
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        st.error("Erro ao decodificar o JSON da resposta da IA. A resposta pode estar mal formatada.")
        st.text_area("Resposta da IA (para depuração):", clean_json_str, height=150)
        return []

def extract_transactions_with_gemini(text: str, file_content: bytes, api_key: str) -> List[Dict[str, Any]]:
    """
    Função universal que usa a IA do Gemini para extrair transações.
    Lida tanto com PDFs de texto quanto com PDFs escaneados (imagens).
    """
    if not api_key:
        st.error("A chave de API do Gemini não foi fornecida.")
        return []
        
    try:
        genai.configure(api_key=api_key)
        
        if not text.strip():
            model = genai.GenerativeModel('gemini-pro-vision')
            prompt = [
                "Você é um especialista em análise de extratos bancários brasileiros e realiza a função de OCR.",
                "Analise as imagens das páginas do extrato a seguir.",
                "Extraia todas as transações, contendo data, descrição e valor.",
                "Retorne os dados como uma lista de objetos JSON. Cada objeto deve ter as chaves 'date' (formato DD/MM/AAAA), 'description' e 'amount' (o valor como um número float, negativo para saídas/débitos).",
                "Ignore saldos, cabeçalhos e qualquer informação que não seja uma transação financeira."
            ]
            
            image_parts = []
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    image_parts.append({"mime_type": "image/png", "data": img_bytes})

            response = model.generate_content(prompt + image_parts)
        
        else:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Você é um especialista em análise de extratos bancários brasileiros. Sua tarefa é extrair transações de um texto de extrato.
            Extraia cada transação e retorne uma lista de objetos JSON com as chaves "date" (DD/MM/AAAA), "description" e "amount" (float, negativo para saídas).
            
            CRÍTICO: Ignore qualquer sequência longa de números que seja claramente um código de barras, ID de documento ou CNPJ, e não um valor monetário. Valores monetários reais quase sempre terão uma vírgula e dois dígitos para os centavos.
            
            Texto do extrato:
            ---
            {text[:15000]}
            ---

            Retorne APENAS a lista de objetos JSON.
            """
            response = model.generate_content(prompt)

        transactions_json = safe_json_parse(response.text)
        if not transactions_json:
            return []
            
        processed_transactions = []
        for t in transactions_json:
            # Ponto da correção: Verifica se 'amount' não é nulo ANTES de processar
            amount_value = t.get('amount')
            if amount_value is None:
                st.warning(f"Transação da IA ignorada por ter valor nulo: {t}")
                continue # Pula para a próxima transação

            try:
                processed_transactions.append({
                    "date": pd.to_datetime(t.get('date'), format='%d/%m/%Y', errors='coerce'),
                    "description": str(t.get('description', '')),
                    "amount": float(amount_value)
                })
            except (ValueError, KeyError, TypeError) as e:
                st.warning(f"Transação da IA ignorada devido a formato inválido: {t} | Erro: {e}")
                continue
                
        return [t for t in processed_transactions if pd.notna(t['date'])]
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API do Gemini: {e}")
        st.error(f"Detalhes: {traceback.format_exc()}")
        return []


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

# --- INTERFACE DA APLICAÇÃO STREAMLIT ---
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
        with st.spinner("Analisando arquivos com a IA do Gemini..."):
            all_transactions = []
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                text = extract_text_from_pdf(file_content)
                
                transactions = extract_transactions_with_gemini(text, file_content, gemini_api_key)
                
                all_transactions.extend(transactions)
                st.sidebar.success(f"Processado {uploaded_file.name}: {len(transactions)} transações")

            if not all_transactions:
                st.error("Nenhuma transação pôde ser extraída de nenhum dos arquivos.")
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
