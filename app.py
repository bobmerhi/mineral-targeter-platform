from fpdf import FPDF

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
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}} - Gerado Automaticamente via SatIntel AI', 0, 0, 'C')
# ========================================================
# 5. REMOTE SENSING TARGET CHANNELS & IBM ENGINE
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")
    
    with st.spinner("Processing multi-spectral analytics..."):
        m_data = fetch_and_calculate_spatz(st.session_state["map_center"], st.session_state["map_center"], selected_year)
    
    st.markdown("#### **WAY 1: Hydrothermal Alteration**")
    w1_c1, w1_c2 = st.columns(2)
    w1_c1.metric("Iron Oxide (Gossans)", m_data["Way_1_Iron_Oxide_Gossan"])
    w1_c2.metric("Clay/Hydroxyl Index", m_data["Way_1_Clay_Phyllic"])
    
    st.markdown("#### **WAY 2: Structural Lineaments**")
    st.metric("Fault Intersection Density", m_data["Way_2_Fault_Density_Index"])
    
    st.markdown("#### **WAY 3: Lithological Silicification**")
    st.metric("Quartz Veining Emissivity", m_data["Way_3_Silica_Flooding_Cap"])
    
    st.markdown("#### **WAY 4: Geobotanical Stress**")
    st.metric("Vegetation Stress Proxy (NDVI)", m_data["Way_4_Geobotanical_Stress"])
    
    st.markdown("#### **WAY 5: GIS Predictive Synthesis**")
    st.metric("WLC Prospectivity Target Score", f"{m_data['Way_5_WLC_Score_Percent']}%")
    st.caption(f"🛰️ Source Pipeline ID: {m_data['Satellite_Used']}")
    st.divider()
    
    if st.button("🚀 Generate 5-Way Geological Synthesis"):
        with st.spinner("O watsonx.ai está correlacionando as matrizes geológicas..."):
            client = get_watsonx_client()
            meta = st.session_state["concession_metadata"]
            
            p1 = "[Role: Geólogo Sénior de Exploração Especialista em Metalogenia de Moçambique]\n"
            p2 = "Execute uma avaliação geológica detalhada para o alvo: " + str(target_commodity) + " nas coordenadas " + str(st.session_state['map_center']) + " para o ano de " + str(selected_year) + ".\n\n"
            p3 = "Dados do Cadastro Mineiro (Trimble Landfolio Moçambique):\n"
            p4 = "- Código da Licença: " + str(meta.get('Código da Licença (Code)', '11521')) + "\n- Nome da Concessão: " + str(meta.get('Nome da Concessão', '')) + "\n- Titular: " + str(meta.get('Titular (Holder Company)', '')) + "\n- Dimensão: " + str(meta.get('Área / Dimensão', '')) + "\n- Validade: " + str(meta.get('Data de Validade (Expiry)', '')) + "\n- Substâncias Registadas: " + str(meta.get('Substâncias', '')) + "\n\n"
            p5 = "Matriz de Telemetria de Detecção Remota (5-Way Model):\n"
            p6 = "- Óxido de Ferro (Gossans): " + str(m_data.get('Way_1_Iron_Oxide_Gossan', 2.4)) + "\n- Índice de Argila/Hidroxilo: " + str(m_data.get('Way_1_Clay_Phyllic', 1.9)) + "\n- Densidade de Falhas Estruturais: " + str(m_data.get('Way_2_Fault_Density_Index', 0.8)) + "\n- Indicador de Silicification: " + str(m_data.get('Way_3_Silica_Flooding_Cap', 0.6)) + "\n- Estresse Geobotânico (NDVI): " + str(m_data.get('Way_4_Geobotanical_Stress', 0.34)) + "\n- Pontuação de Prospectivity Combinada (WLC): " + str(m_data.get('Way_5_WLC_Score_Percent', 88.5)) + "%\n\n"
            p7 = "Directrizes da Tarefa:\n"
            p8 = "Escreva um parecer técnico formal em português. Analise a associação entre o Ouro/Platina e os minerais pegmatíticos listados (Lítio, Turmalinas, Tantalite). Avalie o significado do estresse geobotânico observado e a densidade estrutural. Conclua com recomendações claras de campo (amostragem de solo ou abertura de trincheiras) e um parecer final de 'Perfurar / Não Perfurar' (Drill/No-Drill)."
            
            complete_prompt = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
            
            model = ModelInference(model_id="meta-llama/llama-3-3-70b-instruct", credentials=credentials, project_id=PROJECT_ID)
            st.markdown(model.generate_text(prompt=complete_prompt))
