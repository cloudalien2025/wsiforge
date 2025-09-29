# app.py — WSIForge v0.9.5
import base64, io, json, re, time, zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

APP_TITLE = "WSIForge — Web Story Image Forge"

# Sizes your account supports (from prior API response)
SIZES_SUPPORTED = ["1024x1536", "1536x1024", "1024x1024", "auto"]
DEFAULT_RENDER_SIZE = "1024x1536"  # portrait
DEFAULT_WEBP_QUALITY = 82
MAX_AI_KEYWORDS = 10

# ---------- helpers ----------
def orientation_of_size(sz: str) -> str:
    if sz == "auto": return "auto"
    try:
        w, h = map(int, sz.lower().replace(" ", "").split("x"))
        return "portrait" if h >= w else "landscape"
    except Exception:
        return "portrait"

def normalize_size(requested: str, want: str = "portrait") -> str:
    req = (requested or "").lower().replace(" ", "")
    if req in [s.lower() for s in SIZES_SUPPORTED]:
        return [s for s in SIZES_SUPPORTED if s.lower() == req][0]
    return "1024x1536" if want == "portrait" else "1536x1024"

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

# ---------- OpenAI ----------
@dataclass
class OpenAIClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"

    def generate_image(self, prompt: str, size: str, model: str, timeout: int = 60):
        url = f"{self.base_url}/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "prompt": prompt.strip(), "size": size, "n": 1}
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

def _bytes_from_openai_item(item: dict, timeout: int = 60) -> Optional[bytes]:
    if "b64_json" in item and item["b64_json"]:
        return base64.b64decode(item["b64_json"])
    if "url" in item and item["url"]:
        r = requests.get(item["url"], timeout=timeout)
        r.raise_for_status()
        return r.content
    return None

def _parse_supported_sizes(message: str) -> List[str]:
    m = re.search(r"Supported values are:\s*(.+)", message, flags=re.I)
    if not m: return []
    sizes = re.findall(r"'([\dxauto]+)'", m.group(1))
    return [s for s in sizes if s in {"1024x1024","1024x1536","1536x1024","auto"}]

def _ordered_by_orientation(candidates: List[str], want: str) -> List[str]:
    if want == "portrait":
        order = ["1024x1536","1024x1024","auto","1536x1024"]
    elif want == "landscape":
        order = ["1536x1024","1024x1024","auto","1024x1536"]
    else:
        order = ["1024x1536","1536x1024","1024x1024","auto"]
    out, seen = [], set()
    for s in order:
        if s in candidates and s not in seen:
            out.append(s); seen.add(s)
    for s in candidates:
        if s not in seen: out.append(s)
    return out

def generate_openai_image(
    client: OpenAIClient,
    prompt: str,
    requested_size: str,
    webp_quality: int,
    model: str,
    max_retries: int = 4,
    timeout_s: int = 60,
) -> Tuple[Optional[bytes], Optional[str]]:
    want = orientation_of_size(requested_size)
    size_primary = normalize_size(requested_size, want)
    candidate_sizes = _ordered_by_orientation(SIZES_SUPPORTED, want)
    if size_primary in candidate_sizes:
        candidate_sizes.remove(size_primary)
    candidate_sizes = [size_primary] + candidate_sizes

    delay = 1.2
    last_err = None

    for attempt in range(1, max_retries + 1):
        try_size = candidate_sizes[min(attempt-1, len(candidate_sizes)-1)]
        try:
            resp = client.generate_image(prompt, size=try_size, model=model, timeout=timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("data"): return None, "OpenAI returned no image data."
                raw = _bytes_from_openai_item(data["data"][0], timeout=timeout_s)
                if not raw: return None, "OpenAI returned an empty image payload."
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                webp = to_1080x1920_webp(img, webp_quality)
                return webp, None

            try: j = resp.json()
            except Exception: j = {"error": {"message": resp.text}}
            last_err = j.get("error", {}).get("message", f"HTTP {resp.status_code}")

            if "invalid value" in last_err.lower() and "supported values are" in last_err.lower():
                api_sizes = _parse_supported_sizes(last_err)
                if api_sizes:
                    candidate_sizes = _ordered_by_orientation(api_sizes, want)

            transient = any(x in last_err.lower() for x in ["rate limit","429","timeout","timed out","gateway","overloaded","502","503","504"])
            if transient:
                time.sleep(delay); delay *= 1.8
                continue
            continue

        except Exception as e:
            last_err = str(e)
            transient = any(x in last_err.lower() for x in ["rate limit","429","timeout","timed out","gateway","overloaded","502","503","504"])
            if transient:
                time.sleep(delay); delay *= 1.8
                continue
            continue

    return None, f"OpenAI image request failed after {max_retries} attempt(s): {last_err}"

# ---------- Google Places (Real Photos) ----------
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
    params = {"photoreference": photo_ref, "maxwidth": maxwidth, "key": api_key}
    r = requests.get(GOOGLE_PHOTO, params=params, timeout=timeout, allow_redirects=False)
    if r.status_code in (301, 302) and "Location" in r.headers:
        img_url = r.headers["Location"]
        img_resp = requests.get(img_url, timeout=timeout)
        img_resp.raise_for_status()
        return img_resp.content
    if r.status_code == 200:
        return r.content
    return None

def get_real_photo_candidates(google_key: str, query: str, max_candidates: int = 6, webp_quality: int = DEFAULT_WEBP_QUALITY) -> List[Tuple[str, bytes]]:
    out = []
    try:
        results = google_text_search(google_key, query)
        if not results: return out
        for res in results[:max_candidates*2]:
            pid = res.get("place_id")
            name = res.get("name") or query
            if not pid: continue
            photos = google_place_photos(google_key, pid)
            if not photos: continue
            pref = photos[0].get("photo_reference")
            if not pref: continue
            raw = google_fetch_photo_bytes(google_key, pref)
            if not raw: continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                webp = to_1080x1920_webp(img, webp_quality)
                out.append((name, webp))
            except Exception:
                continue
            if len(out) >= max_candidates: break
    except Exception:
        return out
    return out

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Create 1080×1920 Web Story images from real photos (Google Places) or via OpenAI. Download individually or as a ZIP.")

st.sidebar.header("Mode")
mode = st.sidebar.radio("", options=["Real Photos", "AI Render"], index=1)

st.sidebar.header("Keys")
g_key = st.sidebar.text_input("Google Maps/Places API key", type="password")
st.sidebar.text_input("SerpAPI key (optional)", type="password")  # placeholder
openai_key = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password")

st.sidebar.header("Output")
webp_quality = st.sidebar.slider("WebP quality", 40, 100, DEFAULT_WEBP_QUALITY)

with st.expander("Advanced", expanded=False):
    model = st.selectbox("OpenAI image model", ["gpt-image-1", "dall-e-3"], index=0)
    st.write("Supported sizes:", ", ".join(SIZES_SUPPORTED))

st.subheader("Input")
st.write("Paste keywords (one per line)")
keywords_text = st.text_area("", height=120, placeholder="Vail Colorado cozy restaurant in fall\nVail Village in November\n...")
keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]

st.write("Render base size (OpenAI). We’ll auto-convert to 1080×1920.")
size_choice = st.selectbox("", SIZES_SUPPORTED, index=SIZES_SUPPORTED.index(DEFAULT_RENDER_SIZE),
                           help="Your account supports: 1024x1536 (portrait), 1536x1024 (landscape), 1024x1024, or auto.")

button_label = "Select Candidates" if mode == "Real Photos" else "Generate image(s)"
colA, colB = st.columns([1,1])
with colA: go = st.button(button_label, type="primary")
with colB:
    if st.button("Clear"): st.rerun()

st.markdown("---")
if go:
    if not keywords:
        st.warning("Add at least one keyword.")
    else:
        zip_buffer = io.BytesIO()
        zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)
        previews, errors = [], []

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
                            model=model,
                        )
                        if err:
                            errors.append((kw, err)); st.error(err)
                        else:
                            fname = f"{slugify(kw)}.webp"
                            zf.writestr(fname, webp_bytes)
                            previews.append({"caption": kw, "bytes": webp_bytes, "fname": fname, "mode": "AI"})
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
                            errors.append((kw, msg)); st.warning(msg); continue

                        st.write(f"Select candidates to include for **{kw}**:")
                        chosen = []
                        for label, wb in cands:
                            c1, c2 = st.columns([1,3])
                            with c1:
                                sel = st.checkbox(f"Use: {label}", key=f"sel-{slugify(kw)}-{hash(wb)}", value=True)
                            with c2:
                                st.image(wb, caption=label, use_container_width=True)
                            if sel: chosen.append((label, wb))

                        for j, (label, wb) in enumerate(chosen, start=1):
                            fname = f"{slugify(kw)}-{j}.webp"
                            zf.writestr(fname, wb)
                            previews.append({"caption": f"{kw} #{j}", "bytes": wb, "fname": fname, "mode": "REAL"})
                        st.success(f"Added {len(chosen)} image(s) for {kw}.")

        zf.close(); zip_buffer.seek(0)

        # === Previews (with per-image download for Real Photos) ===
        if previews:
            st.subheader("Previews")
            cols = st.columns(3)
            for idx, item in enumerate(previews):
                with cols[idx % 3]:
                    st.image(item["bytes"], caption=item["caption"], use_container_width=True)
                    if item["mode"] == "REAL":
                        st.download_button(
                            label="Download",
                            data=item["bytes"],
                            file_name=item["fname"],
                            mime="image/webp",
                            key=f"dl-{idx}-{item['fname']}",
                        )

        # Errors (if any)
        if errors:
            st.subheader("Errors")
            for kw, err in errors:
                st.error(f"{kw}: {err}")

        # ZIP download (optional)
        st.subheader("Download")
        st.download_button(
            label="Download all images as ZIP",
            data=zip_buffer.getvalue(),
            file_name="wsiforge_images.zip",
            mime="application/zip",
            key="zip-all",
        )
