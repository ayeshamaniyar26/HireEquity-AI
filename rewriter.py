import os
import logging
from groq import Groq
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def rewrite_job_description(original_jd, flagged_items, style="Inclusive"):
    """
    Rewrites a job description using Groq Llama3 to remove bias with a specified tone style.
    """
    client = get_groq_client()
    if not client:
        raise ValueError("Groq API Key not found. Please add GROQ_API_KEY to your .env file or Streamlit Secrets.")

    # Format the list of flagged phrases for the prompt
    flagged_phrases_str = ", ".join([f"'{item['phrase']}' ({item['category']})" for item in flagged_items])

    prompt = f"""You are an inclusive language expert.
Rewrite this job description to remove all gender bias, age bias, ableism and elitism. These are the flagged phrases to fix: {flagged_phrases_str}

Tone Profile: Rewrite using a '{style}' tone. Follow these style guidelines:
- Inclusive: Highly collaborative, supportive, warm, welcoming, and open.
- Corporate: Professional, structured, formal, objective, and clear.
- Startup: Dynamic, fast-paced, growth-oriented, impact-focused, and energetic.
- FAANG: Scale-focused, highly technical, metric-driven, and innovative.
- Executive: Visionary, strategic, leadership-focused, high-impact, and authoritative.

Rules:
- Keep the same general structure and meaning
- Use gender neutral language
- Replace restrictive experience requirements with reasonable, inclusive equivalents
- Return ONLY the rewritten job description text, no preamble or extra conversational text.
Original JD: {original_jd}"""

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional HR copywriter specialized in writing inclusive, biased-free, and engaging job descriptions."
                },
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3, # lower temperature for structural adherence
            max_tokens=1500
        )
        
        rewritten_jd = response.choices[0].message.content
        return rewritten_jd.strip()

    except Exception as e:
        logger.error(f"Error calling Groq Rewriter API: {e}")
        raise e

