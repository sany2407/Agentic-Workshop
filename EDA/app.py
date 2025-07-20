import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from autogen.agentchat import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_TOKEN = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_TOKEN:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GEMINI_API_TOKEN)

# Global constants
DEFAULT_MODEL = "models/gemini-1.5-flash"
SESSION_DATASET = "uploaded_dataset"
SESSION_PREPROCESSING = "preprocessing_results"
SESSION_ANALYSIS = "analysis_results"
SESSION_FINAL_REPORT = "final_report"

def generate_ai_response(user_prompt, ai_model=DEFAULT_MODEL):
    """Generate response using Gemini AI"""
    try:
        model_instance = genai.GenerativeModel(ai_model)
        response = model_instance.generate_content(user_prompt)
        return response.text
    except Exception as error:
        return f"Error generating response: {str(error)}"

# ===== Custom Agent Classes =====

class DataCleaningSpecialist(AssistantAgent):
    """Agent responsible for data preprocessing and cleaning"""
    
    def generate_reply(self, conversation_history, sender_agent, agent_config=None):
        current_dataset = st.session_state[SESSION_DATASET]
        
        cleaning_prompt = f"""
        You are an expert Data Cleaning Specialist. Analyze the dataset and provide comprehensive preprocessing steps.
        
        **Dataset Overview:**
        - Shape: {current_dataset.shape}
        - Columns: {list(current_dataset.columns)}
        
        **Sample Data:**
        {current_dataset.head(10).to_string()}
        
        **Data Information:**
        {current_dataset.info()}
        
        **Missing Values:**
        {current_dataset.isnull().sum().to_string()}
        
        **Statistical Summary:**
        {current_dataset.describe(include='all').to_string()}
        
        Please provide:
        1. Python code for data cleaning and preprocessing
        2. Explanation of each preprocessing step
        3. Recommendations for handling missing values
        4. Data type optimizations
        
        Format your response with clear code blocks and explanations.
        """
        
        return generate_ai_response(cleaning_prompt)

class DataAnalysisExpert(AssistantAgent):
    """Agent responsible for exploratory data analysis"""
    
    def generate_reply(self, conversation_history, sender_agent, agent_config=None):
        current_dataset = st.session_state[SESSION_DATASET]
        
        analysis_prompt = f"""
        You are a Data Analysis Expert. Perform comprehensive exploratory data analysis.
        
        **Dataset Information:**
        - Records: {len(current_dataset)}
        - Features: {len(current_dataset.columns)}
        - Data types: {current_dataset.dtypes.to_string()}
        
        **Sample Data:**
        {current_dataset.head(8).to_string()}
        
        **Statistical Overview:**
        {current_dataset.describe().to_string()}
        
        Provide detailed analysis including:
        1. 5+ key insights about the dataset
        2. Distribution patterns and trends
        3. Correlation findings (if applicable)
        4. Anomalies or outliers detected
        5. Recommended visualizations with specific chart types
        6. Business implications of findings
        
        Structure your response with clear sections and actionable insights.
        """
        
        return generate_ai_response(analysis_prompt)

class DocumentationGenerator(AssistantAgent):
    """Agent for generating comprehensive reports"""
    
    def generate_reply(self, conversation_history, sender_agent, agent_config=None):
        analysis_findings = st.session_state.get(SESSION_ANALYSIS, "No analysis available")
        
        documentation_prompt = f"""
        You are a Documentation Generator. Create a professional EDA report.
        
        **Analysis Results:**
        {analysis_findings}
        
        Generate a comprehensive report with:
        
        ## Executive Summary
        - Dataset overview and key characteristics
        - Primary findings summary
        
        ## Detailed Findings
        - Statistical insights with interpretation
        - Data quality assessment
        - Pattern identification
        
        ## Visualization Recommendations
        - Specific chart types with justification
        - Key variables to visualize
        - Interactive dashboard suggestions
        
        ## Conclusions and Next Steps
        - Actionable recommendations
        - Areas requiring further investigation
        - Suggested modeling approaches
        
        Use professional language and structure the report for stakeholder consumption.
        """
        
        return generate_ai_response(documentation_prompt)

class QualityAssuranceReviewer(AssistantAgent):
    """Agent for reviewing and validating analysis quality"""
    
    def generate_reply(self, conversation_history, sender_agent, agent_config=None):
        generated_report = st.session_state.get(SESSION_FINAL_REPORT, "No report available")
        
        review_prompt = f"""
        You are a Quality Assurance Reviewer. Evaluate the EDA report for quality and completeness.
        
        **Report to Review:**
        {generated_report}
        
        Assess the following criteria:
        
        **Content Quality (1-10 scale):**
        - Accuracy of statistical interpretations
        - Completeness of analysis coverage
        - Clarity of explanations
        - Actionability of recommendations
        
        **Structure and Presentation:**
        - Logical flow of information
        - Professional formatting
        - Appropriate use of technical terms
        - Stakeholder-friendly language
        
        **Missing Elements:**
        - Important insights overlooked
        - Additional analysis needed
        - Visualization gaps
        
        Provide:
        1. Overall quality score and justification
        2. Specific improvement suggestions
        3. Priority areas for enhancement
        4. Validation of key findings
        """
        
        return generate_ai_response(review_prompt)

class CodeValidationAgent(AssistantAgent):
    """Agent for validating and optimizing generated code"""
    
    def generate_reply(self, conversation_history, sender_agent, agent_config=None):
        preprocessing_code = st.session_state.get(SESSION_PREPROCESSING, "No code available")
        
        validation_prompt = f"""
        You are a Code Validation Agent. Review and optimize the preprocessing code.
        
        **Code to Validate:**
        {preprocessing_code}
        
        Perform comprehensive code review:
        
        **Syntax and Logic:**
        - Check for syntax errors
        - Validate logical flow
        - Identify potential runtime issues
        
        **Best Practices:**
        - Code efficiency optimization
        - Error handling implementation
        - Memory usage considerations
        
        **Improvements:**
        - Suggest alternative approaches
        - Recommend additional preprocessing steps
        - Performance optimization tips
        
        **Output Format:**
        - Validation status (Pass/Fail/Warning)
        - Specific issues identified
        - Corrected code snippets
        - Performance recommendations
        """
        
        return generate_ai_response(validation_prompt)

# ===== System Controller =====
system_controller = UserProxyAgent(
    name="SystemController",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "temp_analysis", "use_docker": False}
)

# ===== Enhanced Streamlit Interface =====

def initialize_streamlit_interface():
    """Setup Streamlit page configuration and styling"""
    st.set_page_config(
        page_title="Advanced Agentic EDA System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sidebar_controls():
    """Create sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ai_model_selection = st.selectbox(
            "Select AI Model",
            ["models/gemini-1.5-flash", "models/gemini-1.5-pro"],
            index=0
        )
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Comprehensive", "Deep"],
            value="Standard"
        )
        
        enable_advanced_features = st.checkbox("Enable Advanced Analytics", value=True)
        
        return ai_model_selection, analysis_depth, enable_advanced_features

def display_dataset_overview(dataset):
    """Display comprehensive dataset overview"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(dataset):,}")
    with col2:
        st.metric("Features", len(dataset.columns))
    with col3:
        st.metric("Missing Values", dataset.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

def main_application():
    """Main application logic"""
    initialize_streamlit_interface()
    
    st.markdown('<h1 class="main-header">🤖 Advanced Agentic EDA System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Multi-Agent Data Analysis Platform powered by Gemini AI</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    selected_model, depth_level, advanced_mode = create_sidebar_controls()
    
    # File upload section
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)",
        type=["csv"],
        help="Select a CSV file to begin automated analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load and store dataset
            dataset = pd.read_csv(uploaded_file)
            st.session_state[SESSION_DATASET] = dataset
            
            # Dataset overview
            st.header("📊 Dataset Overview")
            display_dataset_overview(dataset)
            
            # Data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(dataset.head(20), use_container_width=True)
            
            # Analysis execution
            if st.button("🚀 Execute Multi-Agent Analysis", type="primary"):
                execute_agent_workflow(selected_model, depth_level, advanced_mode)
                
        except Exception as error:
            st.error(f"Error loading dataset: {str(error)}")
    else:
        st.info("👆 Please upload a CSV file to begin analysis")

def execute_agent_workflow(model_name, depth, advanced):
    """Execute the multi-agent analysis workflow"""
    
    progress_container = st.container()
    
    with progress_container:
        analysis_progress = st.progress(0)
        status_text = st.empty()
        
        # Initialize agents
        status_text.text("🔧 Initializing specialized agents...")
        analysis_progress.progress(10)
        
        agent_team = [
            system_controller,
            DataCleaningSpecialist(name="DataCleaner"),
            DataAnalysisExpert(name="AnalysisExpert"),
            DocumentationGenerator(name="ReportGenerator"),
            QualityAssuranceReviewer(name="QAReviewer"),
            CodeValidationAgent(name="CodeValidator"),
        ]
        
        # Setup group communication
        group_chat = GroupChat(agents=agent_team, messages=[])
        chat_manager = GroupChatManager(groupchat=group_chat)
        
        # Execute workflow stages
        execute_preprocessing_stage(agent_team, analysis_progress, status_text)
        execute_analysis_stage(agent_team, analysis_progress, status_text)
        execute_reporting_stage(agent_team, analysis_progress, status_text)
        execute_validation_stage(agent_team, analysis_progress, status_text)
        
        analysis_progress.progress(100)
        status_text.text("✅ Multi-agent analysis completed successfully!")
        
        st.balloons()

def execute_preprocessing_stage(agents, progress, status):
    """Execute data preprocessing stage"""
    status.text("🧹 Executing data preprocessing...")
    progress.progress(25)
    
    preprocessing_result = agents[1].generate_reply([], "SystemController")
    st.session_state[SESSION_PREPROCESSING] = preprocessing_result
    
    with st.expander("🧹 Data Preprocessing Results", expanded=True):
        st.markdown(preprocessing_result)

def execute_analysis_stage(agents, progress, status):
    """Execute data analysis stage"""
    status.text("📈 Performing exploratory data analysis...")
    progress.progress(50)
    
    analysis_result = agents[2].generate_reply([], "SystemController")
    st.session_state[SESSION_ANALYSIS] = analysis_result
    
    with st.expander("📈 Analysis Insights", expanded=True):
        st.markdown(analysis_result)

def execute_reporting_stage(agents, progress, status):
    """Execute report generation stage"""
    status.text("📄 Generating comprehensive report...")
    progress.progress(75)
    
    report_result = agents[3].generate_reply([], "SystemController")
    st.session_state[SESSION_FINAL_REPORT] = report_result
    
    with st.expander("📄 Executive Report", expanded=True):
        st.markdown(report_result)

def execute_validation_stage(agents, progress, status):
    """Execute validation and quality assurance stage"""
    status.text("🔍 Performing quality assurance...")
    progress.progress(90)
    
    # Quality review
    qa_feedback = agents[4].generate_reply([], "SystemController")
    with st.expander("🔍 Quality Assurance Review", expanded=False):
        st.markdown(qa_feedback)
    
    # Code validation
    code_validation = agents[5].generate_reply([], "SystemController")
    with st.expander("⚙️ Code Validation Report", expanded=False):
        st.markdown(code_validation)

if __name__ == "__main__":
    main_application()
