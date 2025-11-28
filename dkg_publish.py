# dkg_publish.py
import os
import requests

def publish_to_dkg(jsonld_str):
    """
    Try to publish JSON-LD to the DKG Edge Node via REST.
    If DKG_EDGE_NODE_URL not defined, raise.
    This function expects the Edge Node to accept JSON-LD POSTs.
    """
    dkg_url = os.getenv("DKG_EDGE_NODE_URL")
    api_key = os.getenv("DKG_API_KEY")
    if not dkg_url:
        raise ValueError("DKG_EDGE_NODE_URL not configured")

    headers = {"Content-Type": "application/ld+json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(dkg_url, data=jsonld_str.encode("utf-8"), headers=headers, timeout=15)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"status": "ok", "raw": resp.text}
