import os
import re
import json
import socket
import threading
import base64
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import streamlit as st
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import our custom backend modules
import wordlists
from bias_detector import (
    scan_wordlist_bias,
    check_semantic_bias,
    calculate_score,
    combine_and_align_flags,
    get_groq_client
)
from jd_generator import generate_job_description
from rewriter import rewrite_job_description
from pdf_export import generate_pdf_report

# =============================================================================
# PERSISTENT BACKGROUND API SERVER STATE
# =============================================================================
# We use Python globals to track the API port and server state across Streamlit runs
if '_server_started' not in globals():
    globals()['_server_started'] = False
if '_server_port' not in globals():
    globals()['_server_port'] = 8585

# =============================================================================
# PERSONA SIMULATION ENGINE (With Groq & Heuristic Fallbacks)
# =============================================================================
def simulate_candidate_personas(jd_text):
    """
    Simulates the reaction and appeal score (0 to 100) for 6 candidate personas.
    Uses Groq Llama 3.1 if API key is present, otherwise falls back to a high-fidelity wordlist heuristic.
    """
    client = get_groq_client()
    if not client:
        # Fallback simulation logic based on wordlist flags
        wordlist_flags = scan_wordlist_bias(jd_text)
        categories = [f["category"].lower() for f in wordlist_flags]
        
        # Default scores
        scores = {
            "Fresh Graduate": 85,
            "Woman": 80,
            "Career Returner": 82,
            "Disabled Candidate": 85,
            "International Candidate": 88,
            "Senior Professional": 85
        }
        
        # Deduct based on scanned categories
        for cat in categories:
            if "gender" in cat:
                scores["Woman"] -= 15
                scores["Career Returner"] -= 10
            if "age" in cat:
                scores["Senior Professional"] -= 20
                scores["Fresh Graduate"] -= 10
            if "elitism" in cat:
                scores["Fresh Graduate"] -= 15
                scores["Career Returner"] -= 10
                scores["International Candidate"] -= 12
            if "ableism" in cat:
                scores["Disabled Candidate"] -= 25
            if "restrictive" in cat:
                scores["Fresh Graduate"] -= 12
                scores["Career Returner"] -= 15
                scores["Senior Professional"] -= 5

        # Bound scores
        for k in scores:
            scores[k] = max(10, min(100, scores[k]))
            
        # Fallback comments generator
        comments = {
            "Fresh Graduate": "The job description seems reasonable, but the requirements seem a bit high for an early-career applicant." if scores["Fresh Graduate"] < 75 else "I find the role description clear and welcoming for early career professionals.",
            "Woman": "I noticed some masculine-coded language that makes the culture seem a bit aggressive and competitive." if scores["Woman"] < 75 else "The language feels highly inclusive and collaborative, which makes me want to apply.",
            "Career Returner": "The strict experience requirements and rigid scheduling suggestions are discouraging." if scores["Career Returner"] < 75 else "The emphasis on capability rather than uninterrupted tenure is reassuring.",
            "Disabled Candidate": "The physical capability assumptions (e.g. standing for long hours) are alienating." if scores["Disabled Candidate"] < 75 else "The description focuses on core deliverables without unnecessary physical requirements.",
            "International Candidate": "The emphasis on local credentials or specific institutional backgrounds is exclusionary." if scores["International Candidate"] < 75 else "The global, open wording makes it clear that candidates of all backgrounds are welcome.",
            "Senior Professional": "The wording implies they are looking for someone younger or with a shorter career path." if scores["Senior Professional"] < 75 else "The scope of the role matches my senior leadership capabilities perfectly."
        }
        
        res = {}
        for k in scores:
            res[k] = {"score": scores[k], "feedback": comments[k]}
        return res
        
    # If client is configured, let's call Groq!
    prompt = f"""Analyze the following job description and simulate the reaction and appeal score (0 to 100) for these 6 candidate personas:
- Fresh Graduate
- Woman
- Career Returner
- Disabled Candidate
- International Candidate
- Senior Professional

Return JSON ONLY. No markdown wrapper, no extra text, format exactly like this:
{{
  "Fresh Graduate": {{"score": 85, "feedback": "Concise feedback here"}},
  "Woman": {{"score": 90, "feedback": "Concise feedback here"}},
  ...
}}

JD Text:
{jd_text}
"""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a DEI dashboard analyst simulating applicant pool reactions based on job description language."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error during Groq persona simulation: {e}")
        # Return fallback if API call fails
        return simulate_candidate_personas("")

# =============================================================================
# LIGHTWEIGHT PYTHON BACKGROUND API SERVER
# =============================================================================
class ApiHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress standard logging in terminal to keep workspace quiet
        return

    def do_POST(self):
        parsed_path = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            payload = json.loads(post_data) if post_data else {}
        except Exception:
            payload = {}

        response_data = {}
        status_code = 200

        # API: Connection Check
        if parsed_path.path == '/api/check_key':
            api_key = os.environ.get("GROQ_API_KEY", "")
            response_data = {"api_key_configured": len(api_key.strip()) > 0}
            
        # API: Generate Job Description
        elif parsed_path.path == '/api/generate':
            role = payload.get("role", "")
            level = payload.get("level", "Mid")
            domain = payload.get("domain", "Tech")
            try:
                jd_text = generate_job_description(role, level, domain)
                response_data = {"jd": jd_text}
            except Exception as e:
                status_code = 500
                response_data = {"error": str(e)}
                
        # API: Audit Job Description
        elif parsed_path.path == '/api/audit':
            jd = payload.get("jd", "")
            try:
                wordlist_flags = scan_wordlist_bias(jd)
                api_key = os.environ.get("GROQ_API_KEY", "")
                semantic_flags = []
                if len(api_key.strip()) > 0:
                    semantic_flags = check_semantic_bias(jd)
                all_flags = combine_and_align_flags(jd, wordlist_flags, semantic_flags)
                score = calculate_score(all_flags)
                response_data = {"score": score, "flagged_items": all_flags}
            except Exception as e:
                status_code = 500
                response_data = {"error": str(e)}

        # API: Rewrite Job Description
        elif parsed_path.path == '/api/rewrite':
            jd = payload.get("jd", "")
            flagged_items = payload.get("flagged_items", [])
            style = payload.get("style", "Inclusive")
            try:
                fixed_jd = rewrite_job_description(jd, flagged_items, style=style)
                response_data = {"fixed_jd": fixed_jd}
            except Exception as e:
                status_code = 500
                response_data = {"error": str(e)}

        # API: Export Report PDF
        elif parsed_path.path == '/api/pdf':
            metadata = payload.get("metadata", {})
            original_score = payload.get("original_score", 100)
            fixed_score = payload.get("fixed_score", 100)
            flagged_items = payload.get("flagged_items", [])
            original_jd = payload.get("original_jd", "")
            fixed_jd = payload.get("fixed_jd", "")
            try:
                pdf_bytes = generate_pdf_report(
                    metadata=metadata,
                    original_score=original_score,
                    fixed_score=fixed_score,
                    flagged_items=flagged_items,
                    original_jd=original_jd,
                    fixed_jd=fixed_jd
                )
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                response_data = {"pdf_base64": pdf_base64}
            except Exception as e:
                status_code = 500
                response_data = {"error": str(e)}

        # API: Simulate Candidate Personas
        elif parsed_path.path == '/api/simulate':
            jd = payload.get("jd", "")
            try:
                simulations = simulate_candidate_personas(jd)
                response_data = simulations
            except Exception as e:
                status_code = 500
                response_data = {"error": str(e)}

        else:
            status_code = 404
            response_data = {"error": "Not Found"}

        # Return HTTP Response
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

# =============================================================================
# SERVER INITIATION GLUE
# =============================================================================
def find_free_port(start_port=8585):
    port = start_port
    while port < 8600:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except socket.error:
                port += 1
    return port

def run_server_thread(port):
    try:
        server = HTTPServer(('localhost', port), ApiHandler)
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")

# If background thread has not been started, spawn it once!
if not globals()['_server_started']:
    free_port = find_free_port(8585)
    globals()['_server_port'] = free_port
    
    server_thread = threading.Thread(target=run_server_thread, args=(free_port,), daemon=True)
    server_thread.start()
    globals()['_server_started'] = True
    logger.info(f"Background API Server started on http://localhost:{free_port}")

# =============================================================================
# STREAMLIT MOUNTING AND VIEWPORT OVERRIDES
# =============================================================================
st.set_page_config(
    page_title="HireEquity Enterprise — AI Bias Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to hide Streamlit UI (header, footer, menu, margins) 
# and stretch the iframe to fill the entire browser viewport
streamlit_override_css = """
<style>
    /* Hide header, footer, options menu, and standard sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stSidebar"] {display: none !important;}
    
    /* Remove padding and margins on page wrapper */
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stApp {
        background-color: transparent !important;
    }

    /* Force the iframe to occupy the full viewport */
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        z-index: 999999 !important;
    }
</style>
"""
st.markdown(streamlit_override_css, unsafe_allow_html=True)

# Read UI Template
ui_file_path = os.path.join(os.path.dirname(__file__), "index.html")
try:
    with open(ui_file_path, "r", encoding="utf-8") as f:
        html_code = f.read()
except Exception as e:
    st.error(f"Failed to load frontend template (index.html): {e}")
    st.stop()

# Inject the dynamic API port into index.html
api_port = globals()['_server_port']
html_code = html_code.replace("const API_PORT = 8585;", f"const API_PORT = {api_port};")

# Display the custom UI inside the full-screen iframe
st.components.v1.html(html_code, height=900)
