# app.py
"""
Truth Delta: AI-Wikipedia vs Wikipedia (Streamlit demo)
Neon Cyber UI version — single-file app.

Replaces Grokipedia/Everipedia with an AI-generated "AI Wikipedia" article.
- Fetches Wikipedia article (python wikipedia library)
- Generates AI encyclopedia article using OpenAI/Groq if configured, otherwise HuggingFace Inference (Gemma)
- Computes similarity (embeddings if available, otherwise token-overlap fallback)
- Uses LLM to extract differences (OpenAI preferred, HF fallback)
- Builds JSON-LD and saves locally
- Optionally publishes to DKG if DKG_EDGE_NODE_URL is set
"""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import wikipedia
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Flexible OpenAI-compatible client import (optional)
openai_client = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optional Hugging Face Inference client fallback
hf_client = None
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# load environment
load_dotenv()

# --------------- Config / Env ---------------
OPENAI_API_KEY = "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"
OPENAI_BASE_URL = "https://api.groq.com/openai/v1"  # e.g., https://api.groq.com/openai/v1
DKG_EDGE_NODE_URL = "https://testnet.origintrail.io/api/publish"
DKG_API_KEY = os.getenv("DKG_API_KEY", "").strip()

# Default model choices (flexible)
if OPENAI_BASE_URL and "groq" in OPENAI_BASE_URL:
    CHAT_MODEL = "llama-3.1-8b-instant"
    EMBEDDING_MODEL = "embed-3-small"
else:
    # When using OpenAI, default to gpt-4o-mini for chat (if available)
    CHAT_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-large"

# Initialize OpenAI-compatible client if API key present
if OPENAI_API_KEY and OpenAI is not None:
    try:
        if OPENAI_BASE_URL:
            openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        else:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

# Initialize HuggingFace Inference client as fallback (no token required for many public models,
# but can use HUGGINGFACEHUB_API_TOKEN if set in env)
if InferenceClient is not None:
    try:
        hf_client = InferenceClient()
    except Exception:
        hf_client = None

# --------------------------- Helper: fetchers ---------------------------
def fetch_wikipedia(topic):
    try:
        wikipedia.set_lang("en")
        page = wikipedia.page(topic)
        return page.content, page.url
    except Exception:
        try:
            results = wikipedia.search(topic)
            if results:
                page = wikipedia.page(results[0])
                return page.content, page.url
            else:
                return f"No Wikipedia page found for '{topic}'", ""
        except Exception as e:
            return f"Could not fetch Wikipedia article: {e}", ""

# --------------------------- Helper: AI-Wikipedia (LLM-generated) ---------------------------
def fetch_ai_wikipedia(topic):
    """
    Generate an AI-style encyclopedia article for `topic`.
    Prefers OpenAI/Groq-compatible client; falls back to HuggingFace Inference.
    Returns (text, url)
    """
    prompt = (
        f"You are an impartial encyclopedia writer. Write a factual, neutral, and well-structured "
        f"encyclopedic article about '{topic}'. Use 5-8 short paragraphs, include key definitions, "
        f"history, and notable facts. Avoid opinion. Keep the tone similar to Wikipedia."
    )

    # Try OpenAI-compatible chat endpoint if available
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an impartial encyclopedia writer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=900,
                temperature=0.0,
            )
            # Normalize common response shapes
            try:
                ai_text = response.choices[0].message.content.strip()
            except Exception:
                ai_text = getattr(response, "text", "") or json.dumps(response)
            ai_url = f"https://ai-wikipedia.local/{topic.replace(' ', '_')}"
            return ai_text, ai_url
        except Exception as e:
            # fall through to HF if OpenAI call fails
            pass

    # Fallback: HuggingFace Inference
    if hf_client:
        try:
            # Use a public instruction model (Gemma recommended if available)
            model_name = "google/gemma-2-9b-it"  # change if you prefer another public model
            # The InferenceClient.text_generation returns different shapes; we handle common ones below
            resp = hf_client.text_generation(model=model_name, inputs=prompt, max_new_tokens=700)
            # resp may be a list or dict depending on SDK version
            if isinstance(resp, str):
                ai_text = resp
            elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
                ai_text = resp[0].get("generated_text") or next(iter(resp[0].values()), "")
            elif isinstance(resp, dict):
                ai_text = resp.get("generated_text") or (resp.get("choices") and resp["choices"][0].get("text")) or str(resp)
            else:
                ai_text = str(resp)
            ai_url = f"https://ai-wikipedia.local/{topic.replace(' ', '_')}"
            return ai_text, ai_url
        except Exception:
            pass

    # If all fails, return a small fallback
    return f"AI Wikipedia generation not available for '{topic}'.", ""

# --------------------------- Helper: embeddings & similarity ---------------------------
def get_embedding(text):
    """
    Return embedding vector as numpy array using OpenAI-compatible client.
    Raises RuntimeError if unavailable.
    """
    if not openai_client:
        raise RuntimeError("OpenAI/Groq client not initialized for embeddings.")
    chunk = text[:3000]
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
    embedding = resp.data[0].embedding
    return np.array(embedding, dtype=float)

def compute_similarity(a_text, b_text):
    """
    Compute similarity between two texts:
    - Use embeddings if openai_client present
    - Otherwise fallback to token-overlap ratio
    Returns float in [0,1]
    """
    try:
        if openai_client:
            e1 = get_embedding(a_text)
            e2 = get_embedding(b_text)
            sim = float(cosine_similarity([e1], [e2])[0][0])
            # clip to [0,1] defensively
            return max(0.0, min(1.0, sim))
    except Exception:
        # ignore and fall back
        pass

    # Token-overlap fallback
    s1 = set(a_text.lower().split()[:1000])
    s2 = set(b_text.lower().split()[:1000])
    if not s1 and not s2:
        return 0.0
    overlap = len(s1 & s2)
    denom = max(1, len(s1) + len(s2))
    return overlap / denom

# --------------------------- Helper: LLM differences extraction ---------------------------
def ask_llm_for_diffs(wiki_text, ai_text, topic):
    """
    Ask an LLM to list up to 6 substantive factual differences where the AI text \
    makes claims not present in Wikipedia, flag hallucinations, and give a way to verify.
    Prefers openai_client; falls back to HF text generation.
    Returns a list of bullet strings.
    """
    prompt = f"""
You are a concise fact-comparison assistant.
Compare the two texts about "{topic}". Do:
1) List up to 6 substantive factual differences or claims that appear in the AI-generated text but not in the Wikipedia text.
2) Flag any statements that seem unsupported or likely hallucinated.
3) For each item provide a one-line sentence suggesting how to verify it (source type).
Output as numbered bullets (1. ...), each bullet should be short (<=2 sentences).
=== WIKIPEDIA ===
{wiki_text[:4000]}
=== AI-WIKIPEDIA ===
{ai_text[:4000]}
"""
    # Use OpenAI-compatible chat if available
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert truth comparison engine."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.0,
            )
            try:
                text = response.choices[0].message.content.strip()
            except Exception:
                text = getattr(response, "text", "") or json.dumps(response)
            # Split into bullets
            bullets = [line.strip(" -•\t\n") for line in text.split("\n") if line.strip()]
            # Re-split if output is a single block with numbered list
            if len(bullets) == 1 and ("\n1." in text or "1)" in text):
                import re
                parts = re.split(r"\n\d+[\).]\s+", text)
                bullets = [p.strip() for p in parts if p.strip()]
            return bullets
        except Exception as e:
            return [f"LLM diff extraction failed (OpenAI): {e}"]

    # Fallback: HuggingFace Inference text generation
    if hf_client:
        try:
            model_name = "google/gemma-2-9b-it"
            resp = hf_client.text_generation(model=model_name, inputs=prompt, max_new_tokens=400)
            if isinstance(resp, str):
                text = resp
            elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
                text = resp[0].get("generated_text") or next(iter(resp[0].values()), "")
            elif isinstance(resp, dict):
                text = resp.get("generated_text") or str(resp)
            else:
                text = str(resp)
            bullets = [line.strip(" -•\t\n") for line in text.split("\n") if line.strip()]
            if len(bullets) == 1 and ("\n1." in text or "1)" in text):
                import re
                parts = re.split(r"\n\d+[\).]\s+", text)
                bullets = [p.strip() for p in parts if p.strip()]
            return bullets
        except Exception as e:
            return [f"LLM diff extraction failed (HF): {e}"]

    return ["No LLM client available to extract differences."]

# --------------------------- Build JSON-LD ---------------------------
def build_jsonld(topic, wiki_url, ai_url, similarity, diffs, meta):
    obj = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "identifier": f"truthdelta:{topic.replace(' ', '_').lower()}:{meta.get('generated_at','')}",
        "name": f"Truth Delta: {topic}",
        "topic": topic,
        "sources": [
            {"name": "Wikipedia", "url": wiki_url},
            {"name": "AI Wikipedia (LLM-generated)", "url": ai_url},
        ],
        "similarityScore": similarity,
        "differences": diffs,
        "provenance": meta
    }
    return json.dumps(obj, indent=2, ensure_ascii=False)

# --------------------------- DKG publish (optional) ---------------------------
def publish_to_dkg(jsonld_str):
    if not DKG_EDGE_NODE_URL:
        raise ValueError("DKG_EDGE_NODE_URL not configured")
    headers = {"Content-Type": "application/ld+json"}
    if DKG_API_KEY:
        headers["Authorization"] = f"Bearer {DKG_API_KEY}"
    resp = requests.post(DKG_EDGE_NODE_URL, data=jsonld_str.encode("utf-8"), headers=headers, timeout=20)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"status": "ok", "text": resp.text}

# --------------------------- Neon Cyber UI ---------------------------
NEON_BG = "#081024"
NEON_PANEL = "#071022"
NEON_ACCENT = "#00f6ff"
NEON_PINK = "#ff007f"
NEON_TEXT = "#E6F7FF"

st.set_page_config(page_title="Truth Delta Scanner", page_icon="🤖", layout="wide")

# Custom CSS for neon look
st.markdown(
    f"""
    <style>
      .stApp {{
          background: linear-gradient(180deg, {NEON_BG} 0%, #001021 100%);
          color: {NEON_TEXT};
      }}
      .neon-card {{
          background: linear-gradient(180deg, rgba(10,10,20,0.55), rgba(6,6,12,0.45));
          border: 1px solid rgba(0,246,255,0.12);
          box-shadow: 0 6px 20px rgba(0,246,255,0.06), 0 0 20px rgba(255,0,127,0.02) inset;
          border-radius: 12px;
          padding: 18px;
          margin-bottom: 18px;
      }}
      .neon-hero {{
          padding: 28px;
          background: linear-gradient(90deg, rgba(0,246,255,0.06), rgba(255,0,127,0.04));
          border-radius: 12px;
          border: 1px solid rgba(0,246,255,0.06);
      }}
      .neon-button {{
          background: linear-gradient(90deg, {NEON_ACCENT}, {NEON_PINK});
          color: #00121a !important;
          font-weight: 700;
          border-radius: 10px;
          padding: 10px 18px;
      }}
      .small-muted {{ color: #91A6B8; font-size:12px; }}
      a.neon-link {{ color: {NEON_ACCENT}; text-decoration: none; font-weight:600; }}
      pre, code {{ background: rgba(0,0,0,0.25); border-radius:8px; padding:8px; color:{NEON_TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header / hero
st.markdown(
    f"""
    <div class="neon-hero neon-card">
      <div style="display:flex; align-items:center; gap:18px;">
        <div style="font-size:46px;"></div>
        <div>
          <div style="font-size:28px; font-weight:700; color:{NEON_ACCENT};">Truth Delta Scanner — Neon</div>
          <div style="color:#9fcbdc; margin-top:6px;">Compare AI-generated encyclopedia vs Wikipedia • detect divergences, hallucinations, and publish verifiable assets</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top-level notices
col_notice_left, col_notice_right = st.columns([3,1])
with col_notice_left:
    if openai_client:
        if OPENAI_BASE_URL and "groq" in OPENAI_BASE_URL:
            st.info("Connected: Groq endpoint. Using Llama-3.1 instant for chat and embeddings.")
        else:
            st.success("Connected: OpenAI-compatible endpoint for chat and embeddings.")
    else:
        if hf_client:
            st.info("OpenAI not configured — falling back to HuggingFace Inference (Gemma) for AI-Wikipedia generation and diff extraction.")
        else:
            st.warning("No LLM client configured. AI generation and diff extraction may not work. Set OPENAI_API_KEY or HUGGINGFACEHUB_API_TOKEN if needed.")

with col_notice_right:
    st.markdown(f"<div style='text-align:right; color:#91A6B8; font-size:13px'>DKG publish: {'enabled' if DKG_EDGE_NODE_URL else 'disabled'}</div>", unsafe_allow_html=True)

st.markdown("")  # spacer

# Input block
with st.container():
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("### 🔎 Analyze a Topic")
    topic = st.text_input("", value="Blockchain", placeholder="Type a topic (e.g., Bitcoin, Climate change)")
    run_col1, run_col2 = st.columns([1,3])
    with run_col1:
        run_btn = st.button("🚀 Run Comparison", key="run1")
    with run_col2:
        st.markdown('<div class="small-muted">Tip: use short topic names for best results in the demo.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    topic = (topic or "").strip()
    if not topic:
        st.error("Please enter a non-empty topic.")
    else:
        # Fetch
        with st.spinner("📡 Fetching sources…"):
            wiki_text, wiki_url = fetch_wikipedia(topic)
            ai_text, ai_url = fetch_ai_wikipedia(topic)

        # Layout: left=preview, right=results
        left, right = st.columns([1, 1.1])

        # LEFT: Previews
        with left:
            st.markdown('<div class="neon-card">', unsafe_allow_html=True)
            st.markdown("### 🌐 Sources")
            st.markdown(f"**Wikipedia:** <a class='neon-link' href='{wiki_url}' target='_blank'>{wiki_url or 'Not found'}</a>", unsafe_allow_html=True)
            st.markdown(f"**AI Wikipedia:** <a class='neon-link' href='{ai_url}' target='_blank'>{ai_url or 'Generated'}</a>", unsafe_allow_html=True)
            st.markdown("### 📄 Article Previews")
            with st.expander("Wikipedia excerpt", expanded=True):
                st.code(wiki_text[:1200] + ("..." if len(wiki_text) > 1200 else ""))
            with st.expander("AI Wikipedia excerpt", expanded=False):
                st.code(ai_text[:1200] + ("..." if len(ai_text) > 1200 else ""))
            st.markdown('</div>', unsafe_allow_html=True)

        # RIGHT: Analysis and JSON-LD
        with right:
            st.markdown('<div class="neon-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Analysis")

            with st.spinner("🧮 Computing semantic similarity…"):
                similarity = compute_similarity(wiki_text or "", ai_text or "")

            st.markdown(f"<div style='font-size:22px; font-weight:700; color:{NEON_ACCENT}'>Similarity/Hallucination</div>", unsafe_allow_html=True)
            st.metric(label="Semantic similarity (0–1)", value=f"{similarity:.3f}")

            st.markdown("### 🧩 Differences & Potential Hallucinations")
            with st.spinner("🔍 Running LLM compare..."):
                diffs = ask_llm_for_diffs(wiki_text or "", ai_text or "", topic)
            for i, d in enumerate(diffs, 1):
                st.markdown(f"**{i}.** {d}")

            st.markdown("### 📦 Knowledge Asset (JSON-LD)")
            meta = {
                "topic": topic,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "tool": "TruthDeltaScanner v0.3 (neon)",
                "model": CHAT_MODEL if openai_client else "hf:google/gemma-2-9b-it" if hf_client else "none",
            }
            jsonld = build_jsonld(topic, wiki_url, ai_url, similarity, diffs, meta)
            st.code(jsonld, language="json")

            # Save button and publish toggle
            st.markdown("")
            save_col, publish_col = st.columns([1,1])
            with save_col:
                safe_topic = topic.replace(" ", "_").lower()
                filename = f"truthdelta_{safe_topic}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jsonld"
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(jsonld)
                    st.success(f"💾 Saved JSON-LD as **{filename}**")
                except Exception as e:
                    st.error(f"Saving failed: {e}")

            with publish_col:
                if DKG_EDGE_NODE_URL:
                    if st.button("⛓️ Publish to DKG", key="publish"):
                        with st.spinner("📡 Publishing to DKG..."):
                            try:
                                resp = publish_to_dkg(jsonld)
                                st.success("🎉 Published to DKG.")
                                st.json(resp)
                            except Exception as e:
                                st.error("Failed to publish to DKG: " + str(e))
                else:
                    st.info("DKG disabled — set DKG_EDGE_NODE_URL in .env to enable publishing.")
            st.markdown('</div>', unsafe_allow_html=True)

# Footer / credits
st.markdown(
    """
    <div style="padding:14px; margin-top:22px; color:#91A6B8; font-size:12px;">
      Built for the OriginTrail x DKG Hackathon • UI: Neon Cyber • Demo only — AI Wikipedia used unless configured otherwise.
    </div>
    """,
    unsafe_allow_html=True,
)
