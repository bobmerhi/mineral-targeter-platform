import streamlit as st
from fpdf import FPDF
from ibm_watsonx_ai.foundation_models import Model

# Page configurations
st.set_page_config(layout="wide", page_title="SatIntel - Mineral Targeter")

# 1. SETUP SYSTEM PROMPT ENGINE FOR LLAMA 3.3
def generate_technical_report(area_ha, data_apresentacao, data_emissao, data_validade, tipo_direito, substancias):
    """Calls watsonx.ai Llama-3-3-70b to generate a structured Portuguese Technical Report."""
    
    system_instruction = (
        "Você é um engenheiro de minas sênior e consultor especialista em regulação mineral em Moçambique. "
        "Sua tarefa é gerar um Parecer Técnico institucional detalhado, formal e rigoroso sobre uma concessão mineira. "
        "O documento deve ser escrito em português formal, utilizando terminologia jurídica e geológica precisa."
    )
    
    user_input = f"""
    Gere um Parecer Técnico estruturado com as seguintes seções claras:
    1. INTRODUÇÃO E ENQUADRAMENTO LEGAL (Análise do direito de exploração)
    2. SUMÁRIO DA ÁREA E DIMENSÕES (Avaliação dos {area_ha} Hectares)
    3. CRONOGRAMA DE VALIDADE E PRAZOS (Análise das datas de apresentação, emissão e expiração)
    4. POTENCIAL GEOLÓGICO E SUBSTÂNCIAS (Análise detalhada das substâncias identificadas: {substancias})
    5. CONCLUSÃO E RECOMENDAÇÕES TÉCNICAS

    Dados da Concessão:
    - Área / Dimensão: {area_ha} Hectares (Ha)
    - Data de Apresentação: {data_apresentacao}
    - Data de Emissão (Concessão): {data_emissao}
    - Data de Validade (Expiry): {data_validade}
    - Tipo de Direito / Estado: {tipo_direito}
    - Substâncias Categoria: {substancias}
    """
    
    # Prompt token injection formatting for Llama 3.3 chat blueprint architecture
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_instruction}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    model_id = "meta-llama/llama-3-3-70b-instruct"
    parameters = {
        "decoding_method": "sample",
        "max_new_tokens": 1200,
        "temperature": 0.3,
        "top_p": 1
    }
    
    # Reassemble authentication properties safely from Streamlit variables
    wml_credentials = {
        "url": "https://ibm.com",
        "apikey": st.secrets["IBM_API_KEY"]
    }
    
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=wml_credentials,
        project_id=st.secrets["PROJECT_ID"]
    )
    
    response = model.generate_text(prompt=full_prompt)
    return response


# 2. INTUITIONAL DOCUMENT PARSER ENGINE (FPDF CLASS DEFINITION)
class TechnicalReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, 'SATINTEL - GEOLOGICAL & MINING INSIGHTS', 0, 1, 'L')
        self.set_draw_color(200, 200, 200)
        self.line(10, 20, 200, 20)
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}} - Gerado Automaticamente via SatIntel API', 0, 0, 'C')

def convert_text_to_pdf(report_content):
    """Compiles the dynamic raw LLM text string payload into safe document byte data."""
    pdf = TechnicalReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    
    # Clean text to safeguard layout generation against standard encoding break exceptions
    safe_text = report_content.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 6, safe_text)
    return pdf.output(dest='S')


# 3. INTERACTIVE PLATFORM APPLICATION LAYOUT
st.title("WAY 5: GIS Predictive Synthesis")

col1, col2 = st.columns([1, 1])

# Left Side Panel - Information Metrics
with col1:
    st.markdown("### Metadata da Concessão")
    
    # Static visual representation of data items matching your design framework
    st.text_input("Company", value="[Nome do Cliente/Titular]", disabled=True)
    area_val = st.text_input("Área / Dimensão (Ha)", value="18,876.81")
    data_ap = st.text_input("Data de Apresentação", value="02/05/2023")
    data_em = st.text_input("Data de Emissão (Concessão)", value="18/06/2025")
    data_val = st.text_input("Data de Validade (Expiry)", value="18/06/2050")
    direito_tipo = st.text_input("Tipo de Direito / Estado", value="Concessão Mineira - Em Vigor")
    subst_list = st.text_area("Substâncias", value="Ouro, Lítio, Esmeralda, Turmalina, Tantalite, Berilo, Espodumena, Lepidolite, Mica, Morganite")

# Right Side Panel - Action Processing Engine
with col2:
    st.metric(label="WLC Prospectivity Target Score", value="87.5%")
    st.caption("Source Pipeline ID: Landsat-Operational-MZ-2024")
    st.write("---")
    
    # Primary Execution Handler tracking explicit internal identity keys to avoid duplicate issues
    if st.button("🚀 Generate 5-Way Geological Synthesis", key="primary_synthesis_runner"):
        with st.spinner("Analisando dados geológicos com Llama 3.3..."):
            try:
                generated_report = generate_technical_report(area_val, data_ap, data_em, data_val, direito_tipo, subst_list)
                st.session_state["report_txt"] = generated_report
            except Exception as error_trace:
                st.error(f"Erro na API watsonx: {error_trace}")

    # Output Management & PDF Delivery Block
    if "report_txt" in st.session_state:
        st.write("---")
        st.markdown("### Parecer Técnico")
        st.write(st.session_state["report_txt"])
        
        # Build document assets asynchronously
        pdf_bytes = convert_text_to_pdf(st.session_state["report_txt"])
        
        st.download_button(
            label="📥 Download Parecer Técnico (PDF)",
            data=pdf_bytes,
            file_name="Parecer_Tecnico_SatIntel.pdf",
            mime="application/pdf",
            key="pdf_export_download_trigger"
        )
