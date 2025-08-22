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
                
                # Log para debug - mostra quantos caracteres foram extraídos de cada página
                if len(page_text.strip()) > 0:
                    st.sidebar.write(f"Página {page_num}: {len(page_text)} caracteres extraídos")
                else:
                    st.sidebar.warning(f"Página {page_num}: Nenhum texto extraído (pode ser um PDF escaneado)")
                
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return ""
    
    # Log para debug - mostra o total de caracteres extraídos
    if len(full_text.strip()) > 0:
        st.sidebar.info(f"Total de texto extraído: {len(full_text)} caracteres")
    else:
        st.sidebar.error("Nenhum texto pôde ser extraído do PDF. Pode ser um PDF escaneado (imagem).")
    
    return full_text

def parse_amount(amount_str: str) -> float:
    """Converte uma string de valor monetário para float."""
    if not isinstance(amount_str, str):
        return 0.0
    
    # Remove caracteres não numéricos exceto vírgula, ponto e sinal negativo
    cleaned_str = re.sub(r'[^\d,\-\.]', '', str(amount_str))
    
    # Verifica se é negativo
    is_negative = '-' in cleaned_str
    cleaned_str = cleaned_str.replace('-', '')
    
    # Verifica se tem tanto ponto quanto vírgula (provavelmente ponto como separador de milhar)
    if '.' in cleaned_str and ',' in cleaned_str:
        # Remove pontos (separadores de milhar) e substitui vírgula por ponto
        cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
    elif ',' in cleaned_str:
        # Substitui vírgula por ponto (separador decimal)
        cleaned_str = cleaned_str.replace(',', '.')
    
    try:
        value = float(cleaned_str)
        return -value if is_negative else value
    except (ValueError, TypeError):
        return 0.0

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
            
            transactions.append({
                "date": pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce'),
                "description": description,
                "amount": parse_amount(amount_str)
            })
    
    return [t for t in transactions if pd.notna(t['date'])]

def parse_santander(text: str) -> List[Dict[str, Any]]:
    """Parser para extratos do Santander."""
    transactions = []
    
    lines = text.split('\n')
    
    # Padrão Santander: data, descrição e valor
    date_pattern = r'^\d{2}/\d{2}/\d{4}'
    amount_pattern = r'(-?\d{1,3}(?:\.\d{3})*,\d{2})$'
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Verifica se a linha começa com uma data
        if re.match(date_pattern, line):
            parts = line.split()
            if len(parts) >= 3:
                date_str = parts[0]
                
                # Encontra o valor no final da linha
                amount_match = re.search(amount_pattern, line)
                if amount_match:
                    amount_str = amount_match.group(1)
                    description = line[len(date_str):-len(amount_str)].strip()
                    
                    # Ignora linhas de saldo
                    if any(term in description.upper() for term in ['SALDO', 'S A L D O', 'EXTRATO', 'SALDO ANTERIOR', 'SALDO DO DIA']):
                        continue
                    
                    transactions.append({
                        "date": pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce'),
                        "description": description,
                        "amount": parse_amount(amount_str)
                    })
    
    return [t for t in transactions if pd.notna(t['date'])]

def parse_inter(text: str) -> List[Dict[str, Any]]:
    """Parser para extratos do Banco Inter."""
    transactions = []
    current_date = None
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Verifica se é uma linha de data
        date_match = re.search(r'(\d{1,2} de [a-zA-Zç]+ de \d{4})', line, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(1)
            try:
                month_map = {
                    'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 
                    'abril': 'April', 'maio': 'May', 'junho': 'June', 
                    'julho': 'July', 'agosto': 'August', 'setembro': 'September', 
                    'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'
                }
                
                for pt, en in month_map.items():
                    if pt in date_str.lower():
                        date_str = date_str.replace(pt, en)
                        break
                
                current_date = datetime.strptime(date_str, '%d de %B de %Y')
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
    """Tenta analisar JSON de forma segura, mesmo com strings malformadas."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.warning(f"Erro ao analisar JSON: {e}")
        # Tenta encontrar e corrigir strings não escapadas
        try:
            # Tenta encontrar o array JSON na resposta
            json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Corrige aspas não escapadas
                json_str = re.sub(r'(?<!\\)"', r'\"', json_str)
                return json.loads(json_str)
        except:
            pass
        
        # Última tentativa: extrair manualmente os objetos
        try:
            objects = []
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, json_str, re.DOTALL)
            for match in matches:
                try:
                    obj = json.loads(match)
                    if all(key in obj for key in ['date', 'description', 'amount']):
                        objects.append(obj)
                except:
                    continue
            return objects
        except:
            st.error("Não foi possível extrair transações do JSON retornado pelo Gemini.")
            return []

def parse_with_gemini(text: str, api_key: str) -> List[Dict[str, Any]]:
    """Usa a API do Gemini para extrair transações de um texto de extrato."""
    if not api_key:
        st.error("A chave de API do Gemini não foi fornecida.")
        return []
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limita o texto para evitar exceder o limite de tokens
        text_sample = text[:8000] if len(text) > 8000 else text
        
        prompt = f"""
        Você é um especialista em análise de dados financeiros. Sua tarefa é extrair transações de um texto de extrato bancário.
        O texto a seguir é o conteúdo de um extrato em PDF. Identifique cada transação e retorne uma lista de objetos JSON.
        Cada objeto deve ter EXATAMENTE as seguintes chaves: "date" (no formato "DD/MM/AAAA"), "description" (a descrição completa) e "amount" (o valor como um número, usando ponto como separador decimal, e negativo para saídas).
        Ignore linhas de saldo, cabeçalhos ou qualquer outra informação que não seja uma transação.

        Texto do extrato:
        ---
        {text_sample}
        ---

        Retorne APENAS a lista de objetos JSON, sans formatação markdown ou texto adicional. Exemplo de saída:
        [
          {{"date": "30/06/2025", "description": "PIX TRANSF BRUNO C28/06", "amount": -1500.00}},
          {{"date": "30/06/2025", "description": "SISPAG PIX H2 ESTACIONAMENTO...", "amount": 1500.00}}
        ]
        """
        
        # Adiciona timeout e tratamento de erro para a chamada da API
        try:
            start_time = time.time()
            response = model.generate_content(prompt)
            processing_time = time.time() - start_time
            st.sidebar.info(f"Gemini processou em {processing_time:.1f} segundos")
        except Exception as api_error:
            st.error(f"Erro na chamada da API do Gemini: {api_error}")
            return []
            
        cleaned_response = response.text.strip()
        
        # Remove marcações de código se presentes
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        
        # Tenta parsear a resposta como JSON
        transactions_json = safe_json_parse(cleaned_response)
        
        if not transactions_json:
            st.error("Não foi possível extrair transações da resposta do Gemini.")
            return []
        
        transactions = []
        for t in transactions_json:
            try:
                transactions.append({
                    "date": pd.to_datetime(t['date'], format='%d/%m/%Y', errors='coerce'),
                    "description": t['description'],
                    "amount": float(t['amount'])
                })
            except (ValueError, KeyError) as e:
                st.warning(f"Transação ignorada devido a erro: {e}")
                continue
        
        return [t for t in transactions if pd.notna(t['date'])]
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar ou processar a resposta da API do Gemini: {e}")
        st.error(f"Detalhes do erro: {traceback.format_exc()}")
        return []

def detect_bank_and_parse(text: str, filename: str, gemini_key: str) -> List[Dict[str, Any]]:
    """Detecta o banco, tenta o parser normal e usa Gemini como fallback."""
    nfkd_form = unicodedata.normalize('NFKD', text.lower())
    normalized_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    parser = None
    bank_name = "Desconhecido"
    
    if 'itau' in normalized_text or 'itáu' in normalized_text or 'uniclass' in normalized_text:
        bank_name = "Itaú"
        parser = parse_itau
    elif 'santander' in normalized_text:
        bank_name = "Santander"
        parser = parse_santander
    elif 'inter' in normalized_text or 'banco inter' in normalized_text:
        bank_name = "Banco Inter"
        parser = parse_inter
    elif 'bradesco' in normalized_text:
        bank_name = "Bradesco"
    elif 'caixa' in normalized_text or 'cef' in normalized_text:
        bank_name = "Caixa Econômica Federal"
    elif 'nubank' in normalized_text or 'nu bank' in normalized_text:
        bank_name = "Nubank"
    
    st.sidebar.info(f"Arquivo '{filename}' identificado como: {bank_name}")
    
    transactions = []
    if parser:
        transactions = parser(text)
        st.sidebar.info(f"Parser específico encontrou {len(transactions)} transações")

    # Usa Gemini se não encontrou transações suficientes ou se não tem parser específico
    if (len(transactions) < 3 and gemini_key) or (not parser and gemini_key):
        st.sidebar.warning(f"Usando análise com Gemini AI para '{filename}'...")
        with st.spinner("A IA do Gemini está analisando o extrato..."):
            gemini_transactions = parse_with_gemini(text, gemini_key)
            if gemini_transactions:
                transactions = gemini_transactions
                st.sidebar.info(f"Gemini encontrou {len(transactions)} transações")
            else:
                st.sidebar.error("Gemini não conseguiu extrair transações.")
    
    return transactions

def categorize_transaction(description: str) -> str:
    """Categoriza uma transação com base em palavras-chave na descrição."""
    if not isinstance(description, str):
        return 'Outros'
        
    desc_lower = description.lower()
    rules = {
        'Receitas': ['pix recebido', 'sispag', 'salário', 'credito', 'deposito', 'transferencia recebida', 'ted recebida'],
        'Alimentação': ['ifood', 'restaurante', 'mercado', 'supermercado', 'lanche', 'padaria', 'açai', 'pizzaria', 'hamburguer'],
        'Moradia': ['cemig', 'dmae', 'aluguel', 'condominio', 'claro', 'telefonica', 'vivo', 'tim', 'oi', 'energia', 'agua', 'internet'],
        'Transporte': ['uber', 'posto', 'gasolina', 'estacionamento', 'localiza', 'onibus', 'taxi', 'metro', 'uber', '99', 'combustivel'],
        'Compras': ['lojas', 'shopping', 'mercado pag', 'havan', 'leroy', 'amazon', 'magazine', 'centauro', 'renner', 'riachuelo'],
        'Saúde': ['farmacia', 'drogaria', 'unimed', 'hospital', 'clinica', 'medico', 'dentista', 'laboratorio', 'academia'],
        'Serviços & Taxas': ['pagamento fatura', 'juros', 'iof', 'seguro', 'boleto', 'crediario', 'tarifa', 'anuidade', 'taxa'],
        'Educação': ['escola', 'faculdade', 'universidade', 'curso', 'livraria', 'material escolar'],
        'Lazer': ['cinema', 'netflix', 'spotify', 'viagem', 'hotel', 'passagem'],
    }
    
    for category, keywords in rules.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return 'Outros'

# --- INTERFACE DA APLICAÇÃO STREAMLIT ---

# Inicialização do estado da sessão
if 'excluded_ids' not in st.session_state:
    st.session_state.excluded_ids = set()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Verifica se a chave de API está configurada
if "gemini_api_key" not in st.secrets:
    st.error("A chave de API do Gemini não foi encontrada nos segredos do Streamlit. Por favor, adicione-a.")
    st.info("Acesse a página de configurações do seu app e configure o 'gemini_api_key'.")
    st.stop()

gemini_api_key = st.secrets["gemini_api_key"]

with st.sidebar:
    st.header("Controles")
    st.success("Chave de API do Gemini carregada com sucesso!")
    
    uploaded_files = st.file_uploader(
        "Selecione os arquivos PDF",
        type="pdf",
        accept_multiple_files=True
    )
    
    filter_term = st.text_input(
        "Desconsiderar Titular (por nome):",
        help="Digite um nome para remover transações internas da análise."
    )

if uploaded_files:
    current_filenames = [f.name for f in uploaded_files]
    
    if 'df_original' not in st.session_state or st.session_state.get('processed_files') != current_filenames:
        with st.spinner("Processando arquivos... Isso pode levar alguns instantes."):
            all_transactions = []
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                text = extract_text_from_pdf(file_content)
                
                if not text or len(text.strip()) < 50:
                    st.error(f"O arquivo {uploaded_file.name} parece estar vazio ou não pôde ser lido corretamente.")
                    # Tenta usar o Gemini mesmo com texto vazio (pode ser PDF escaneado)
                    if gemini_api_key:
                        st.sidebar.warning("Tentando usar Gemini para PDF possivelmente escaneado...")
                        with st.spinner("A IA do Gemini está analisando o extrato escaneado..."):
                            gemini_transactions = parse_with_gemini("Extrato bancário em PDF escaneado", gemini_api_key)
                            if gemini_transactions:
                                all_transactions.extend(gemini_transactions)
                                st.sidebar.info(f"Gemini encontrou {len(gemini_transactions)} transações no PDF escaneado")
                    continue
                
                transactions = detect_bank_and_parse(text, uploaded_file.name, gemini_api_key)
                all_transactions.extend(transactions)
                st.sidebar.success(f"Processado {uploaded_file.name}: {len(transactions)} transações")

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
        if len(removed_by_name) > 0:
            st.sidebar.info(f"{len(removed_by_name)} transações removidas pelo filtro de nome.")

    if st.session_state.excluded_ids:
        df_processed = df_processed[~df_processed['id'].isin(st.session_state.excluded_ids)]

    st.header("Análise Financeira")

    # Cálculos de métricas
    if not df_processed.empty:
        total_income = df_processed[df_processed['amount'] > 0]['amount'].sum()
        total_expenses = df_processed[df_processed['amount'] < 0]['amount'].sum()
        net_balance = total_income + total_expenses  # total_expenses já é negativo
        
        # Calcula meses analisados
        if 'date' in df_processed.columns and not df_processed['date'].empty:
            unique_months = df_processed['date'].dt.to_period('M').nunique()
            months_analyzed = max(unique_months, 1)  # Pelo menos 1 mês
        else:
            months_analyzed = 1
            
        average_income = total_income / months_analyzed if months_analyzed > 0 else 0
        presumed_income = average_income * 0.30

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entradas (Total)", f"R$ {total_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col2.metric("Saídas (Total)", f"R$ {abs(total_expenses):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col3.metric("Saldo Líquido", f"R$ {net_balance:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col4.metric("Meses Analisados", months_analyzed)

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Ticket Médio / Mês", f"R$ {average_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        col6.metric("Capacidade 30%", f"R$ {presumed_income:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        
        # Análise por categoria
        st.markdown("---")
        st.subheader("Análise por Categoria")
        
        category_expenses = df_processed[df_processed['amount'] < 0].groupby('category')['amount'].sum().abs()
        category_income = df_processed[df_processed['amount'] > 0].groupby('category')['amount'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Despesas por Categoria**")
            if not category_expenses.empty:
                for category, amount in category_expenses.items():
                    st.write(f"{category}: R$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
            else:
                st.write("Nenhuma despesa categorizada")
        
        with col2:
            st.write("**Receitas por Categoria**")
            if not category_income.empty:
                for category, amount in category_income.items():
                    st.write(f"{category}: R$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
            else:
                st.write("Nenhuma receita categorizada")

    st.markdown("---")
    
    st.subheader("Resumo Mensal")
    if not df_processed.empty and 'date' in df_processed.columns:
        df_processed['Mês'] = df_processed['date'].dt.strftime('%Y-%m')
        monthly_summary = df_processed.groupby('Mês').apply(lambda x: pd.Series({
            'Entradas': x[x['amount'] > 0]['amount'].sum(),
            'Saídas': x[x['amount'] < 0]['amount'].sum(),
            'Saldo': x['amount'].sum()
        })).reset_index()
        
        # Formata os valores para exibição
        formatted_summary = monthly_summary.copy()
        for col_name in ['Entradas', 'Saídas', 'Saldo']:
            formatted_summary[col_name] = formatted_summary[col_name].apply(
                lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            )
        
        st.dataframe(formatted_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Não há dados para exibir o resumo mensal.")

    st.markdown("---")

    st.subheader("Transações Identificadas")
    
    if not df_processed.empty:
        df_for_editor = df_processed.copy()
        df_for_editor['Data'] = df_for_editor['date'].dt.strftime('%d/%m/%Y')
        df_for_editor['Valor (R$)'] = df_for_editor['amount'].apply(
            lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        )
        df_for_editor.rename(columns={'description': 'Descrição', 'category': 'Categoria'}, inplace=True)

        # Mostra o número total de transações
        st.info(f"Total de transações identificadas: {len(df_for_editor)}")
        
        # Configuração para mostrar todas as linhas
        st.markdown("**Visualização completa de todas as transações:**")
        
        # Exibe todas as transações em uma tabela
        st.dataframe(
            df_for_editor[['Data', 'Descrição', 'Valor (R$)', 'Categoria']],
            use_container_width=True,
            hide_index=True,
            height=min(800, 35 * len(df_for_editor) + 38)  # Ajusta a altura automaticamente
        )
        
        # Usamos um formulário para agrupar a seleção e o botão
        with st.form("selection_form"):
            st.markdown("**Selecionar transações para desconsiderar:**")
            
            # Cria uma lista de opções para seleção
            options = []
            for idx, row in df_for_editor.iterrows():
                option_label = f"{row['Data']} - {row['Descrição']} - {row['Valor (R$)']}"
                options.append((row['id'], option_label))
            
            # Widget de multiselect para selecionar transações
            selected_ids = st.multiselect(
                "Selecione as transações para desconsiderar:",
                options=[opt[1] for opt in options],
                format_func=lambda x: x
            )
            
            submitted = st.form_submit_button("Desconsiderar Transação(ões) Selecionada(s)")
            if submitted and selected_ids:
                # Mapeia as seleções de volta para os IDs
                selected_option_ids = []
                for selected in selected_ids:
                    for opt_id, opt_label in options:
                        if opt_label == selected:
                            selected_option_ids.append(opt_id)
                            break
                
                st.session_state.excluded_ids.update(selected_option_ids)
                st.rerun()
            elif submitted:
                st.warning("Nenhuma transação foi selecionada.")
    else:
        st.info("Nenhuma transação encontrada para exibir.")

else:
    st.info("Aguardando o upload de arquivos PDF para iniciar a análise.")
