# app.py  — WSIForge v0.9.1 (patched)
import base64
import io
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

# =========================
# ---- Constants/Config ----
# =========================
APP_TITLE = "WSIForge — Web Story Image Forge"
PORTRAIT = "1024x1792"
LANDSCAPE = "1792x1024"
SQUARE = "1024x1024"
ALLOWED_SIZES = [PORTRAIT, LANDSCAPE, SQUARE]
DEFAULT_RENDER_SIZE = PORTRAIT  # best fit for 1080x1920 conversion
DEFAULT_WEBP_QUALITY = 82
MAX_AI_KEYWORDS = 10

# =========================
# ---- Utilities ----------
# =========================
def normalize_size(requested: str) -> str:
    req = (requested or "").lower().replace(" ", "")
    if req in [s.lower() for s in ALLOWED_SIZES]:
        return [s for s in ALLOWED_SIZES if s.lower() == req][0]
    # heuristics
    try:
        w, h = map(int, req.split("x"))
        if h >= w:
            return PORTRAIT
        return LANDSCAPE
    except Exception:
        return SQUARE

def to_1080x1920_webp(img: Image.Image, webp_quality: int = DEFAULT_WEBP_QUALITY) -> bytes:
    target_w, target_h = 1080, 1920
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_size = (int(src_w * scale), int(src_h * scale))
    img = img.resize(new_size, Image.LANCZOS)
    left = (img.width - target_w) // 2
    top = (img.height - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=int(webp_quality), method=6)
    return buf.getvalue()

def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.strip().lower()).strip("-") or "image"

def b64img(b: bytes) -> str:
    return "data:image/webp;base64," + base64.b64encode(b).decode("utf-8")

# =========================
# ---- OpenAI (Images) ----
# =========================
@dataclass
class OpenAIClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"

    def generate_image(
        self, prompt: str, size: str, model: str = "gpt-image-1", timeout: int = 60
    ):
        url = f"{self.base_url}/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "prompt": prompt.strip(),
            "size": size,
            "response_format": "b64_json",
            "quality": "high",
        }
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

def generate_openai_image(
    client: OpenAIClient,
    prompt: str,
    requested_size: str,
    webp_quality: int = DEFAULT_WEBP_QUALITY,
    model: str = "gpt-image-1",
    max_retries: int = 4,
    timeout_s: int = 60,
) -> Tuple[Optional[bytes], Optional[str]]:
    size = normalize_size(requested_size)
    delay = 1.2
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.generate_image(prompt, size=size, model=model, timeout=timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("data"):
                    return None, "OpenAI returned no image data."
                b64 = data["data"][0].get("b64_json")
                if not b64:
                    return None, "OpenAI returned empty image payload."
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                webp = to_1080x1920_webp(img, webp_quality)
                return webp, None

            # Non-200
            try:
                j = resp.json()
            except Exception:
                j = {"error": {"message": resp.text}}
            last_err = j.get("error", {}).get("message", f"HTTP {resp.status_code}")
            transient = any(x in str(last_err).lower() for x in [
                "rate limit", "429", "timeout", "timed out",
                "gateway", "overloaded", "502", "503", "504"
            ])
            if attempt < max_retries and transient:
                time.sleep(delay); delay *= 1.8; continue
            break

        except Exception as e:
            last_err = str(e)
            transient = any(x in last_err.lower() for x in [
                "rate limit", "429", "timeout", "timed out",
                "gateway", "overloaded", "502", "503", "504"
            ])
            if attempt < max_retries and transient:
                time.sleep(delay); delay *= 1.8; continue
            break

    hint = ""
    if normalize_size(requested_size) != requested_size:
        hint = " Tip: choose 1024x1792 for portrait Web Stories."
    return None, f"OpenAI image request failed after {max_retries} attempt(s): {last_err}.{hint}"

# =========================
# ---- Google Places ------
# =========================
GOOGLE_TEXTSEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_DETAILS = "https://maps.googleapis.com/maps/api/place/details/json"
GOOGLE_PHOTO = "https://maps.googleapis.com/maps/api/place/photo"

def google_text_search(api_key: str, query: str, timeout: int = 20) -> List[dict]:
    params = {"query": query, "key": api_key}
    r = requests.get(GOOGLE_TEXTSEARCH, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("results", [])

def google_place_photos(api_key: str, place_id: str, timeout: int = 20) -> List[dict]:
    params = {"place_id": place_id, "fields": "photo", "key": api_key}
    r = requests.get(GOOGLE_DETAILS, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("result", {}).get("photos", []) or []

def google_fetch_photo_bytes(api_key: str, photo_ref: str, maxwidth: int = 1600, timeout: int = 60) -> Optional[bytes]:
    # First request returns a redirect to the actual image URL.
    params = {"photoreference": photo_ref, "maxwidth": maxwidth, "key": api_key}
    r = requests.get(GOOGLE_PHOTO, params=params, timeout=timeout, allow_redirects=False)
    if r.status_code in (301, 302) and "Location" in r.headers:
        img_url = r.headers["Location"]
        img_resp = requests.get(img_url, timeout=timeout)
        img_resp.raise_for_status()
        return img_resp.content
    # Some environments auto-follow; fallback:
    if r.status_code == 200:
        return r.content
    return None

def get_real_photo_candidates(google_key: str, query: str, max_candidates: int = 6, webp_quality: int = DEFAULT_WEBP_QUALITY) -> List[Tuple[str, bytes]]:
    """
    Returns list of (label, webp_bytes) candidates.
    """
    out = []
    try:
        results = google_text_search(google_key, query)
        if not results:
            return out
        # Take top results and pull 1 photo from each
        for res in results[:max_candidates*2]:  # oversample, some places have no photos
            pid = res.get("place_id")
            name = res.get("name") or query
            if not pid:
                continue
            photos = google_place_photos(google_key, pid)
            if not photos:
                continue
            pref = photos[0].get("photo_reference")
            if not pref:
                continue
            raw = google_fetch_photo_bytes(google_key, pref)
            if not raw:
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                webp = to_1080x1920_webp(img, webp_quality)
                out.append((name, webp))
            except Exception:
                continue
            if len(out) >= max_candidates:
                break
    except Exception:
        return out
    return out

# =========================
# ---- Streamlit UI -------
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Create 1080×1920 Web Story images from real photos (Google Places) or via OpenAI. Download everything as a ZIP.")

# Sidebar
st.sidebar.header("Mode")
mode = st.sidebar.radio("",
                        options=["Real Photos", "AI Render"],
                        index=1 if "render_mode" not in st.session_state else (0 if st.session_state.get("render_mode")=="Real Photos" else 1))
st.session_state["render_mode"] = mode

st.sidebar.header("Keys")
g_key = st.sidebar.text_input("Google Maps/Places API key", type="password")
serp_key = st.sidebar.text_input("SerpAPI key (optional)", type="password")
openai_key = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password")

st.sidebar.header("Output")
webp_quality = st.sidebar.slider("WebP quality", min_value=40, max_value=100, value=DEFAULT_WEBP_QUALITY)

# Main controls
st.subheader("Input")
st.write("Paste keywords (one per line)")
keywords_text = st.text_area("", height=120, placeholder="Vail Village in November\nSkiing in Blue Sky Basin\n...")
keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]

st.write("Render base size (OpenAI). We’ll auto-convert to 1080×1920.")
size_choice = st.selectbox("", ALLOWED_SIZES, index=ALLOWED_SIZES.index(DEFAULT_RENDER_SIZE),
                           help="Supported sizes: 1024x1024, 1024x1792 (portrait), 1792x1024 (landscape).")

colA, colB = st.columns([1,1])
with colA:
    go = st.button("Generate image(s)", type="primary")
with colB:
    clear = st.button("Clear")

if clear:
    st.experimental_set_query_params()  # benign reset
    st.rerun()

# Work area
st.markdown("---")
if go:
    if not keywords:
        st.warning("Add at least one keyword.")
    else:
        import zipfile
        from io import BytesIO

        zip_buffer = BytesIO()
        zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

        previews = []
        errors = []

        if mode == "AI Render":
            if not openai_key:
                st.error("OpenAI API key is required for AI Render.")
            else:
                if len(keywords) > MAX_AI_KEYWORDS:
                    st.info(f"You entered {len(keywords)} keywords. Only the first {MAX_AI_KEYWORDS} will be generated to avoid rate limits.")
                    keywords = keywords[:MAX_AI_KEYWORDS]

                client = OpenAIClient(api_key=openai_key)
                for i, kw in enumerate(keywords, start=1):
                    with st.status(f"Generating {i}/{len(keywords)} — {kw}", expanded=False):
                        webp_bytes, err = generate_openai_image(
                            client=client,
                            prompt=kw,
                            requested_size=size_choice,
                            webp_quality=webp_quality,
                        )
                        if err:
                            errors.append((kw, err))
                            st.error(err)
                        else:
                            fname = f"{slugify(kw)}.webp"
                            zf.writestr(fname, webp_bytes)
                            previews.append((kw, webp_bytes))
                            st.success(f"Done: {fname}")

        else:  # Real Photos
            if not g_key:
                st.error("Google Maps/Places API key is required for Real Photos.")
            else:
                for i, kw in enumerate(keywords, start=1):
                    with st.status(f"Searching photos {i}/{len(keywords)} — {kw}", expanded=False):
                        cands = get_real_photo_candidates(g_key, kw, max_candidates=6, webp_quality=webp_quality)
                        if not cands:
                            msg = "No Google photo candidates found."
                            errors.append((kw, msg))
                            st.warning(msg)
                            continue

                        # Selection UI
                        st.write(f"Select candidates to include for **{kw}**:")
                        chosen = []
                        for label, wb in cands:
                            c1, c2 = st.columns([1,3])
                            with c1:
                                sel = st.checkbox(f"Use: {label}", key=f"{kw}-{label}-{hash(wb)}", value=True)
                            with c2:
                                st.image(wb, caption=label, use_column_width=True)
                            if sel:
                                chosen.append((label, wb))

                        for j, (_, wb) in enumerate(chosen, start=1):
                            fname = f"{slugify(kw)}-{j}.webp"
                            zf.writestr(fname, wb)
                            previews.append((f"{kw} #{j}", wb))
                        st.success(f"Added {len(chosen)} image(s) for {kw}.")

        zf.close()
        zip_buffer.seek(0)

        # Previews grid
        if previews:
            st.subheader("Previews")
            cols = st.columns(3)
            for idx, (cap, wb) in enumerate(previews):
                with cols[idx % 3]:
                    st.image(wb, caption=cap, use_column_width=True)

        # Errors, if any
        if errors:
            st.subheader("Errors")
            for kw, err in errors:
                st.error(f"{kw}: {err}")

        # Download ZIP
        st.subheader("Download")
        st.download_button(
            label="Download all images as ZIP",
            data=zip_buffer.getvalue(),
            file_name="wsiforge_images.zip",
            mime="application/zip",
        )

# Helper info
with st.expander("Tips & Notes", expanded=False):
    st.markdown(
        """
- **OpenAI sizes allowed**: 1024x1024, 1024x1792 (portrait), 1792x1024 (landscape).  
  If you choose something else, we normalize under the hood.
- All outputs are **converted to 1080×1920 WebP** for Web Stories.
- On transient OpenAI errors (429/5xx/timeouts), we **retry with exponential backoff**.
- **Real Photos** uses Google Places Text Search → Place Photos. Candidate selection lets you pick which to include.
        """
    )
