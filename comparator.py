# comparator.py
import requests
import wikipedia
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Helper: fetch Wikipedia article text and URL
def fetch_wikipedia(topic):
    try:
        wikipedia.set_lang("en")
        page = wikipedia.page(topic)
        return page.content, page.url
    except Exception as e:
        # fallback: search then try first result
        try:
            results = wikipedia.search(topic)
            if results:
                page = wikipedia.page(results[0])
                return page.content, page.url
        except Exception:
            pass
        return f"Could not fetch Wikipedia article for topic: {topic}. Error: {e}", ""

# Helper: fetch Grokipedia text - if not available, mock using Wikipedia summary/transform
def fetch_grokipedia(topic):
    grok_api = os.getenv("GROKIPEDIA_API_URL")
    if grok_api:
        try:
            r = requests.get(f"{grok_api}/article", params={"q": topic}, timeout=10)
            r.raise_for_status()
            data = r.json()
            # assumes data has 'content' and 'url'
            return data.get("content", ""), data.get("url", grok_api)
        except Exception as e:
            # fallback to mock
            pass
    # MOCK: get truncated wikipedia summary and slightly transform it for demo
    try:
        summary = wikipedia.summary(topic, sentences=6)
        # make small transformation to simulate AI rewrite
        grok_text = summary.replace("is", "is (GrokAI)")[:4000]
        grok_url = f"https://grokipedia.example/{topic.replace(' ', '_')}"
        return grok_text, grok_url
    except Exception:
        return "Grokipedia content not available (mock placeholder)", ""

# Compute similarity using OpenAI embeddings (if OPENAI_API_KEY set) else simple TF fallback
def get_embedding_openai(text, model="text-embedding-3-large"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing for embeddings.")
    openai.api_key = api_key
    response = openai.Embedding.create(input=text, model=model)
    emb = response["data"][0]["embedding"]
    return np.array(emb, dtype=float)

def compute_similarity_and_diffs(wiki_text, grok_text, topic, openai_key=None):
    # Compute embeddings similarity (try OpenAI embeddings, else fallback to simple bag-of-words)
    try:
        emb1 = get_embedding_openai(wiki_text[:2000])
        emb2 = get_embedding_openai(grok_text[:2000])
        sim = float(cosine_similarity([emb1], [emb2])[0][0])
    except Exception:
        # fallback: simple token overlap ratio
        s1 = set(wiki_text.lower().split()[:1000])
        s2 = set(grok_text.lower().split()[:1000])
        overlap = len(s1 & s2)
        union = max(1, len(s1) + len(s2))
        sim = overlap / union

    # Use LLM to get human-readable differences if key available
    diffs = []
    if openai_key:
        openai.api_key = openai_key
        prompt = build_diff_prompt(wiki_text, grok_text, topic)
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini", # or "gpt-4o" if available
                messages=[{"role":"system","content":"You are a concise fact-comparison assistant."},
                          {"role":"user","content":prompt}],
                max_tokens=400,
                temperature=0.0
            )
            llm_out = resp["choices"][0]["message"]["content"].strip()
            # split into bullets
            diffs = [line.strip("-• \n\t") for line in llm_out.split("\n") if line.strip()]
        except Exception as e:
            diffs = [f"LLM diff extraction failed: {e}"]
    else:
        diffs = ["OpenAI key not configured — LLM differences not available."]

    return sim, diffs

def build_diff_prompt(wiki_text, grok_text, topic):
    # craft a short but precise prompt
    prompt = f"""
Compare the following two texts about "{topic}" and:
1) List up to 6 substantive factual differences where Grokipedia claims diverge or add new claims compared to Wikipedia.
2) Flag any statements that appear to be unsupported or likely hallucinated.
3) Provide a one-line suggestion for evidence to check for each flagged item.

=== WIKIPEDIA ===
{wiki_text[:4000]}

=== GROKIPEDIA ===
{grok_text[:4000]}

Output as numbered bullets, each: Difference / why / suggestion.
"""
    return prompt

# Build JSON-LD string
def build_jsonld(topic, wiki_url, grok_url, similarity, diffs, metadata):
    obj = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "identifier": f"truthdelta:{topic.replace(' ', '_').lower()}:{metadata.get('generated_at','')}",
        "name": f"Truth Delta: {topic}",
        "topic": topic,
        "sources": [
            {"name": "Wikipedia", "url": wiki_url},
            {"name": "Grokipedia", "url": grok_url}
        ],
        "similarityScore": similarity,
        "differences": diffs,
        "provenance": metadata
    }
    return json.dumps(obj, indent=2, ensure_ascii=False)
