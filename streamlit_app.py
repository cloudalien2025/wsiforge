# streamlit_app.py
# WSIForge v0.9 — Web Story Image Forge
# Real-photos (Google Places / Street View / optional SerpAPI) or AI rendered (OpenAI) → 1080×1920 WebP + ZIP download
# Layout and controls intentionally mirror ImageForge.

import io
import os
import time
import math
import json
import base64
import zipfile
import random
import requests
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageOps
import streamlit as st

# ------------------------------- #
# --------- CONSTANTS ----------- #
# ------------------------------- #

APP_NAME = "WSIForge v0.9"
WEB_STORY_SIZE = (1080, 1920)  # (w, h)
DEFAULT_WEBP_QUALITY = 82
OPENAI_IMAGE_MODEL = "gpt-image-1"

# ------------------------------- #
# --------- UTILITIES ----------- #
# ------------------------------- #

def slugify(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in text)
    out = "-".join([s for s in out.split("-") if s])
    return out[:120] if out else "image"

def smart_fit_to_story(img: Image.Image, target=(1080, 1920)) -> Image.Image:
    """Center-crop to target aspect then resize (keeps most salient content)."""
    tw, th = target
    target_ratio = tw / th
    w, h = img.size
    if w == 0 or h == 0:
        return img.convert("RGB").resize(target, Image.LANCZOS)

    src_ratio = w / h
    if src_ratio > target_ratio:
        # too wide → crop width
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, h))
    else:
        # too tall → crop height
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        img = img.crop((0, y0, w, y0 + new_h))

    return img.convert("RGB").resize(target, Image.LANCZOS)

def to_webp_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()

def download_file(url: str, timeout=15) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r.content
    except Exception:
        pass
    return None

def backoff_sleep(i: int):
    time.sleep(min(1.5 * (2 ** i) + random.random(), 10))

# ------------------------------- #
# ----- GOOGLE PLACES / SV ------ #
# ------------------------------- #

def google_text_search(query: str, api_key: str) -> Optional[dict]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.ok:
            data = r.json()
            if data.get("results"):
                return data["results"][0]
    except Exception:
        return None
    return None

def google_place_details(place_id: str, api_key: str) -> Optional[dict]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": api_key,
        "fields": "name,geometry,photo"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.ok:
            return r.json().get("result")
    except Exception:
        return None
    return None

def google_photo_urls(detail: dict, api_key: str, max_photos=6, maxwidth=1600) -> List[str]:
    urls = []
    photos = detail.get("photos") or []
    for p in photos[:max_photos]:
        ref = p.get("photo_reference")
        if not ref: 
            continue
        # build Places Photos endpoint
        u = (
            "https://maps.googleapis.com/maps/api/place/photo"
            f"?maxwidth={maxwidth}&photo_reference={ref}&key={api_key}"
        )
        urls.append(u)
    return urls

def street_view_url(lat: float, lng: float, heading=None, fov=80, pitch=5, size="640x640", api_key=""):
    # We use metadata to confirm pano availability before fetching
    base = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{lat},{lng}",
        "fov": fov,
        "pitch": pitch,
        "key": api_key
    }
    if heading is not None:
        params["heading"] = heading
    return requests.Request("GET", base, params=params).prepare().url

def street_view_has_pano(lat: float, lng: float, radius_m: int, api_key: str) -> Optional[Tuple[float,float]]:
    meta = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "radius": radius_m, "key": api_key}
    try:
        r = requests.get(meta, params=params, timeout=12)
        if r.ok:
            d = r.json()
            if d.get("status") == "OK":
                loc = d.get("location", {})
                return loc.get("lat"), loc.get("lng")
    except Exception:
        pass
    return None

# ------------------------------- #
# --------- SERPAPI (opt) ------- #
# ------------------------------- #

def serpapi_image_urls(query: str, serp_key: str, num=4) -> List[str]:
    urls = []
    base = "https://serpapi.com/search.json"
    params = {"engine": "google_images", "q": query, "num": max(1, min(num, 10)), "api_key": serp_key}
    try:
        r = requests.get(base, params=params, timeout=20)
        if r.ok:
            data = r.json()
            for item in data.get("images_results", [])[:num]:
                u = item.get("original") or item.get("thumbnail")
                if u:
                    urls.append(u)
    except Exception:
        pass
    return urls

# ------------------------------- #
# -------- OPENAI IMAGES -------- #
# ------------------------------- #

def openai_generate_image(prompt: str, api_key: str, size=(1080,1920), tries=4, timeout=60) -> Optional[bytes]:
    """Robust wrapper around OpenAI Images (gpt-image-1). Returns raw bytes (PNG) or None."""
    w, h = size
    size_str = f"{w}x{h}"
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "size": size_str,
        "n": 1,
    }

    for i in range(tries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if not data.get("data"):
                    return None
                item = data["data"][0]
                if "b64_json" in item and item["b64_json"]:
                    return base64.b64decode(item["b64_json"])
                if "url" in item and item["url"]:
                    raw = download_file(item["url"], timeout=timeout)
                    if raw:
                        return raw
                # sometimes different casing or structure; last-chance scan:
                for d in data.get("data", []):
                    for k, v in d.items():
                        if isinstance(v, str) and v.startswith("http"):
                            raw = download_file(v, timeout=timeout)
                            if raw:
                                return raw
                        if k.endswith("b64") or k.endswith("b64_json"):
                            try:
                                raw = base64.b64decode(v)
                                if raw:
                                    return raw
                            except Exception:
                                pass
            elif r.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(i)
                continue
            else:
                # transient HTML (Cloudflare) errors: retry
                txt = r.text.lower()
                if "cloudflare" in txt or "gateway" in txt:
                    backoff_sleep(i)
                    continue
                # otherwise give up
                return None
        except requests.exceptions.ReadTimeout:
            backoff_sleep(i)
        except Exception:
            backoff_sleep(i)
    return None

# ------------------------------- #
#  SESSION STATE STRUCTURE/HELP  #
# ------------------------------- #

def ensure_state():
    ss = st.session_state
    ss.setdefault("candidates", {})   # {keyword: [ {thumb: bytes, source, title, index, chosen: bool} ]}
    ss.setdefault("outputs", [])      # [(filename, bytes)]
    ss.setdefault("last_zip", None)   # (zip_bytes, zip_name)

def add_output(name: str, data: bytes):
    name = name if name.endswith(".webp") else f"{name}.webp"
    st.session_state["outputs"].append((name, data))

def make_zip(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn, data in files:
            z.writestr(fn, data)
    buf.seek(0)
    return buf.read()

# ------------------------------- #
# --------- UI SECTIONS --------- #
# ------------------------------- #

st.set_page_config(page_title=APP_NAME, page_icon="🎞️", layout="wide")
ensure_state()

st.sidebar.title("Mode")
mode = st.sidebar.radio("",
                        ["Real Photos", "AI Render"],
                        index=0,
                        help="Real Photos uses Google Places Photos + Street View (optional SerpAPI). AI Render uses OpenAI to generate images.",
                        label_visibility="collapsed")

st.sidebar.title("Keys")
gmap_key = st.sidebar.text_input("Google Maps/Places API key", type="password", help="Enable Places API + Street View Static API.")
serp_key = st.sidebar.text_input("SerpAPI key (optional)", type="password", help="Used for extra reference candidates.")
openai_key = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password")

st.sidebar.title("Output")
webp_quality = st.sidebar.slider("WebP quality", 60, 100, DEFAULT_WEBP_QUALITY)

st.sidebar.title("Sources to use")
use_places = st.sidebar.checkbox("Google Places Photos", value=True, help="Fetch official place photos")
use_street = st.sidebar.checkbox("Google Street View", value=True, help="Fetch Street View pano near the place")
use_serp = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", value=False, help="Adds extra web images (if key is set)")

st.sidebar.title("Street View")
sv_radius = st.sidebar.slider("Search radius (meters)", 50, 500, 250)

st.title(APP_NAME)
st.caption("Create **1080×1920 Web Story** images from real photos (Google Places / Street View / optional SerpAPI) or via OpenAI. Download everything as a ZIP.")

# Input area
keywords = st.text_area("Paste keywords (one per line)", height=130,
                        placeholder="e.g. Tavern on the Square, Vail Colorado\nBlue Moose Pizza, Vail")
st.caption("Up to 10 keywords for AI Render. For Real Photos, each keyword is searched as a place.")

# Render base (for AI; we always export 1080×1920 WebP anyway)
st.selectbox("Render base size (OpenAI)", 
             options=["1024x1024", "1024x1536", "1536x1024", "auto"],
             index=2, key="ai_size", help="OpenAI base render; we still export 1080×1920.")

colA, colB = st.columns([1, 0.3])
with colA:
    if mode == "Real Photos":
        action_label = "Select candidates"
    else:
        action_label = "Generate image(s)"
    go = st.button(action_label, type="primary")
with colB:
    clr = st.button("Clear")

if clr:
    st.session_state["candidates"].clear()
    st.session_state["outputs"].clear()
    st.session_state["last_zip"] = None
    st.rerun()

# ------------------------------- #
# --------- REAL PHOTOS ----------#
# ------------------------------- #
if go and mode == "Real Photos":
    st.session_state["outputs"].clear()
    st.session_state["last_zip"] = None

    if not gmap_key:
        st.error("Google Maps/Places API key is required for Real Photos.")
    else:
        kws = [k.strip() for k in keywords.splitlines() if k.strip()]
        if not kws:
            st.warning("Please enter at least one keyword.")
        else:
            with st.spinner("Collecting candidates…"):
                for kw in kws:
                    place = google_text_search(kw, gmap_key)
                    if not place:
                        st.warning(f"No place found for '{kw}'.")
                        continue

                    detail = google_place_details(place.get("place_id", ""), gmap_key) or {}
                    photo_urls = google_photo_urls(detail, gmap_key, max_photos=8)

                    candidates = []
                    # Places Photos
                    if use_places:
                        for idx, u in enumerate(photo_urls):
                            raw = download_file(u)
                            if not raw:
                                continue
                            try:
                                img = Image.open(io.BytesIO(raw)).convert("RGB")
                                thumb = img.copy()
                                thumb.thumbnail((400, 400))
                                tb = to_webp_bytes(thumb, 80)
                                candidates.append({
                                    "title": f"Google Places Photo — {detail.get('name', kw)}",
                                    "source": "Google Places",
                                    "thumb": tb,
                                    "raw": raw,
                                    "key": f"gp-{idx}"
                                })
                            except Exception:
                                continue

                    # Street View
                    if use_street and place.get("geometry"):
                        loc = place["geometry"]["location"]
                        sv_loc = street_view_has_pano(loc["lat"], loc["lng"], sv_radius, gmap_key)
                        if sv_loc:
                            sv_url = street_view_url(sv_loc[0], sv_loc[1], size="640x640", api_key=gmap_key)
                            raw = download_file(sv_url)
                            if raw:
                                try:
                                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                                    thumb = img.copy()
                                    thumb.thumbnail((400, 400))
                                    tb = to_webp_bytes(thumb, 80)
                                    candidates.append({
                                        "title": f"Google Street View — {detail.get('name', kw)}",
                                        "source": "Street View",
                                        "thumb": tb,
                                        "raw": raw,
                                        "key": "sv-0"
                                    })
                                except Exception:
                                    pass

                    # SerpAPI (optional)
                    if use_serp and serp_key:
                        urls = serpapi_image_urls(kw, serp_key, num=4)
                        for sidx, u in enumerate(urls):
                            raw = download_file(u)
                            if not raw:
                                continue
                            try:
                                img = Image.open(io.BytesIO(raw)).convert("RGB")
                                thumb = img.copy()
                                thumb.thumbnail((400, 400))
                                tb = to_webp_bytes(thumb, 80)
                                candidates.append({
                                    "title": f"SerpAPI (Google Images) — {kw}",
                                    "source": "SerpAPI",
                                    "thumb": tb,
                                    "raw": raw,
                                    "key": f"sa-{sidx}"
                                })
                            except Exception:
                                continue

                    st.session_state["candidates"][kw] = candidates

# Show candidates when in Real Photos
if mode == "Real Photos" and st.session_state["candidates"]:
    st.subheader("Candidates — select one or check **Select all** for each keyword, then **Create Web Story image(s)**")

    for kw, cands in st.session_state["candidates"].items():
        st.markdown(f"### {kw}")
        if not cands:
            st.info("No candidates found for this keyword.")
            continue

        # Select all toggle per keyword
        sel_col1, sel_col2 = st.columns([0.2, 1])
        with sel_col1:
            sel_all = st.checkbox("Select all", key=f"selall-{slugify(kw)}", value=True)

        selected_keys = []
        grid_cols = st.columns(3)
        for idx, cand in enumerate(cands):
            col = grid_cols[idx % 3]
            with col:
                st.image(cand["thumb"], use_column_width=True, caption=f"**{cand['title']}**  \n*{cand['source']}*")
                ck = st.checkbox("Use", key=f"use-{slugify(kw)}-{cand['key']}", value=sel_all)
                if ck:
                    selected_keys.append(cand["key"])

        def create_for_keyword(keys: List[str]):
            count = 0
            for cand in cands:
                if cand["key"] not in keys:
                    continue
                try:
                    img = Image.open(io.BytesIO(cand["raw"]))
                    story = smart_fit_to_story(img, WEB_STORY_SIZE)
                    fname = slugify(kw)
                    # disambiguate filenames when selecting many:
                    if count > 0:
                        fname = f"{fname}-{count+1}"
                    add_output(fname + ".webp", to_webp_bytes(story, webp_quality))
                    count += 1
                except Exception as e:
                    st.warning(f"Failed on one candidate: {e}")

            if count:
                st.success(f"Created {count} Web Story image(s) for **{kw}**.")

        # Button to create images from selected candidates
        btn_area = st.container()
        if btn_area.button("Create Web Story image(s)", key=f"make-{slugify(kw)}"):
            if not selected_keys:
                st.warning("Pick at least one candidate.")
            else:
                create_for_keyword(selected_keys)

# ------------------------------- #
# ----------- AI RENDER ----------#
# ------------------------------- #
if go and mode == "AI Render":
    st.session_state["outputs"].clear()
    st.session_state["last_zip"] = None

    if not openai_key:
        st.error("OpenAI API key is required for AI Render.")
    else:
        kws = [k.strip() for k in keywords.splitlines() if k.strip()]
        if not kws:
            st.warning("Add 1–10 keywords (one per line).")
        elif len(kws) > 10:
            st.warning("AI Render allows up to 10 keywords. Only the first 10 will be used.")
            kws = kws[:10]

        size_choice = st.session_state.get("ai_size", "1536x1024")
        if size_choice == "auto":
            base_size = (1536, 1024)
        else:
            try:
                w, h = map(int, size_choice.split("x"))
                base_size = (w, h)
            except Exception:
                base_size = (1536, 1024)

        for i, kw in enumerate(kws, 1):
            st.markdown(f"### {i}/{len(kws)} — {kw}")
            with st.spinner("Contacting OpenAI…"):
                raw = openai_generate_image(kw, openai_key, size=base_size)
            if not raw:
                st.error("OpenAI did not return an image (rate limit/timeout/gateway). Try again.")
                continue

            try:
                base_img = Image.open(io.BytesIO(raw))
            except Exception:
                st.error("OpenAI returned data that could not be read as an image.")
                continue

            story = smart_fit_to_story(base_img, WEB_STORY_SIZE)
            out_bytes = to_webp_bytes(story, webp_quality)
            fname = slugify(kw) + ".webp"
            add_output(fname, out_bytes)
            st.image(out_bytes, caption=f"{fname}", use_column_width=True)

# ------------------------------- #
# -------- ZIP & DOWNLOAD --------#
# ------------------------------- #

if st.session_state["outputs"]:
    st.subheader("Downloads")
    n = len(st.session_state["outputs"])
    st.write(f"{n} image(s) ready.")
    # Build ZIP on demand
    if st.button("Build ZIP"):
        zip_name = f"{slugify(APP_NAME)}-{int(time.time())}.zip"
        zip_bytes = make_zip(st.session_state["outputs"])
        st.session_state["last_zip"] = (zip_bytes, zip_name)

    if st.session_state["last_zip"]:
        zb, zn = st.session_state["last_zip"]
        st.download_button("Download ZIP", data=zb, file_name=zn, mime="application/zip")

    # Also allow individual downloads
    with st.expander("Individual images"):
        for fn, data in st.session_state["outputs"]:
            st.download_button(f"Download {fn}", data=data, file_name=fn, mime="image/webp")
