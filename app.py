import os
import re
import html
import random
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
FAVICON_PATH = Path(__file__).with_name("netflix_favicon.svg")
if not FAVICON_PATH.exists():
    FAVICON_PATH.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="10" fill="#080808"/><path d="M13 8h12l14 32V8h12v48H39L25 24v32H13z" fill="#e50914"/></svg>',
        encoding="utf-8",
    )

st.set_page_config(
    page_title="Netflix Insights",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "netflix_titles.csv"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
    TMDB_ACCESS_TOKEN = st.secrets.get("TMDB_API_READ_ACCESS_TOKEN", "")
except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
    TMDB_ACCESS_TOKEN = os.getenv("TMDB_API_READ_ACCESS_TOKEN", "")

# ============================================================
# 3. CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    :root { --netflix-red:#e50914; --netflix-dark:#080808; --netflix-card:#181818; --netflix-text:#f5f5f5; --netflix-muted:#a3a3a3; }
    html, body, [class*="css"] { font-family:'Inter',sans-serif; }
    .stApp { background:#080808; color:#f5f5f5; }
    [data-testid="stHeader"] { background:rgba(8,8,8,.95); }
    [data-testid="stSidebar"] { background:#111; border-right:1px solid #242424; }
    [data-testid="stSidebar"] * { color:#f5f5f5; }
    h1,h2,h3,h4 { color:#fff !important; letter-spacing:-.5px; }
    .brand { color:#e50914; font-size:34px; font-weight:900; letter-spacing:-1.5px; margin-bottom:0; }
    .brand-subtitle { color:#a3a3a3; font-size:11px; letter-spacing:3px; margin-top:-5px; margin-bottom:28px; }
    .hero { position:relative; overflow:hidden; min-height:330px; padding:clamp(28px,5vw,60px); border-radius:14px; background:radial-gradient(circle at 85% 20%,rgba(229,9,20,.24),transparent 30%),linear-gradient(105deg,rgba(0,0,0,.98),rgba(10,10,10,.91) 55%,rgba(20,5,7,.62)),linear-gradient(135deg,#33080b,#111 70%); border:1px solid #282828; display:flex; flex-direction:column; justify-content:center; margin-bottom:28px; box-shadow:0 22px 70px rgba(0,0,0,.22); }
    .hero:after { content:"N"; position:absolute; right:5%; top:50%; transform:translateY(-50%); font-size:clamp(180px,28vw,360px); line-height:1; font-weight:900; letter-spacing:-.12em; color:rgba(229,9,20,.13); pointer-events:none; z-index:0; text-shadow:0 0 70px rgba(229,9,20,.10); }
    .hero > * { position:relative; z-index:1; }
    .hero-tag { color:#ff5159; font-size:12px; font-weight:800; letter-spacing:3px; margin-bottom:14px; }
    .hero h1 { font-size:clamp(34px,5vw,58px); line-height:1.08; font-weight:900; max-width:750px; margin-bottom:14px; }
    .hero p { color:#d1d1d1; font-size:15px; max-width:680px; line-height:1.8; }
    .analytics-toolbar { background:linear-gradient(145deg,#191919,#101010); border:1px solid #333; border-radius:14px; padding:18px 18px 8px; margin:12px 0 20px; }
    .insight-box { background:linear-gradient(135deg,rgba(229,9,20,.12),rgba(25,25,25,.95)); border:1px solid rgba(229,9,20,.28); border-left:4px solid #e50914; border-radius:9px; padding:12px 15px; margin:8px 0 22px; color:#ddd; font-size:13px; line-height:1.65; }
    .section-title { font-size:23px; font-weight:800; margin-top:26px; margin-bottom:5px; }
    .section-subtitle { color:#a3a3a3; font-size:13px; margin-bottom:20px; }
    .poster-card { background:linear-gradient(145deg,#1c1c1c,#111); border-radius:14px; overflow:hidden; border:1px solid rgba(255,255,255,.07); margin-bottom:12px; transition:transform .3s,border-color .3s,box-shadow .3s; box-shadow:0 5px 18px rgba(0,0,0,.18); height:100%; }
    .poster-card:hover { transform:translateY(-5px); border-color:rgba(229,9,20,.8); box-shadow:0 12px 30px rgba(229,9,20,.12); }
    .poster-image-container { position:relative; width:100%; aspect-ratio:2/3; overflow:hidden; background:linear-gradient(145deg,#292929,#111); }
    .poster-image { display:block; width:100%; height:100%; object-fit:cover; transition:transform .45s,filter .3s; }
    .poster-card:hover .poster-image { transform:scale(1.04); filter:brightness(.8); }
    .poster-fallback { position:absolute; inset:0; display:flex; flex-direction:column; justify-content:center; align-items:center; gap:12px; color:#a3a3a3; font-size:10px; font-weight:700; letter-spacing:1.5px; text-align:center; padding:12px; background:radial-gradient(circle at top,#383838,transparent 65%),linear-gradient(145deg,#222,#101010); }
    .fallback-icon { font-size:42px; }
    .poster-image-container:not(.poster-missing) .poster-fallback { display:none; }
    .poster-image-container.poster-missing .poster-fallback { display:flex; }
    .poster-image-container.poster-missing .poster-overlay { display:none; }
    .poster-overlay { position:absolute; inset:auto 0 0 0; display:flex; justify-content:space-between; align-items:center; gap:8px; padding:30px 10px 10px; color:#fff; font-size:11px; font-weight:700; background:linear-gradient(transparent,rgba(0,0,0,.85)); pointer-events:none; }
    .overlay-type { background:#e50914; border-radius:4px; padding:4px 7px; text-transform:uppercase; }
    .overlay-year { color:#e5e5e5; }
    .poster-info { padding:13px 12px 15px; min-height:100px; }
    .poster-title { color:#fff; font-size:14px; font-weight:800; line-height:1.45; margin-bottom:7px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; overflow-wrap:anywhere; }
    .poster-meta { color:#a8a8a8; font-size:11px; line-height:1.6; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
    .badge { display:inline-block; background:#303030; color:#fff; border-radius:4px; padding:4px 7px; font-size:10px; font-weight:700; margin-bottom:8px; }
    .detail-panel { background:#181818; padding:24px; border-radius:12px; border:1px solid #303030; margin:20px 0 25px; }
    .detail-description { color:#d1d1d1; line-height:1.9; font-size:14px; }
    .stat-card { background:#181818; border:1px solid #292929; border-radius:10px; padding:19px; min-height:105px; }
    .stat-label { color:#a3a3a3; font-size:12px; margin-bottom:8px; }
    .stat-value { color:#fff; font-size:27px; font-weight:900; }
    .stat-accent { color:#e50914; }
    .empty-state { text-align:center; padding:65px 20px; background:#141414; border:1px dashed #414141; border-radius:12px; }
    .empty-state p { color:#a3a3a3; }
    .footer { border-top:1px solid #292929; color:#777; font-size:11px; text-align:center; padding:25px 10px; margin-top:45px; }
    div.stButton > button { width:100%; border-radius:9px; font-weight:700; min-height:40px; border:1px solid #333; background:#252525; color:#fff; transition:all .2s; white-space:normal; }
    div.stButton > button:hover { border-color:#e50914; color:#fff; background:#333; }
    div.stButton > button[kind="primary"] { background:#e50914; border-color:#e50914; color:#fff; }
    div.stButton > button[kind="primary"]:hover { background:#b80710; border-color:#b80710; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#1c1c1c,#111); border:1px solid #303030; border-radius:10px; padding:15px; box-shadow:0 8px 24px rgba(0,0,0,.14); }
    div[data-testid="stMetricLabel"] { color:#b3b3b3; }
    div[data-testid="stMetricValue"] { color:#fff; }
    [data-testid="stPlotlyChart"] { background:linear-gradient(145deg,#151515,#0e0e0e); border:1px solid #282828; border-radius:14px; padding:8px; }
    [data-testid="stDataFrame"] { border:1px solid #303030; border-radius:12px; overflow:hidden; }
    [data-testid="stSidebar"] [role="radiogroup"] label { border:1px solid transparent; border-radius:10px; padding:9px 12px; margin:3px 0; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#202020; border-color:#393939; }
    hr { border-color:#292929; }
    @media(max-width:768px) { .poster-card { border-radius:10px; } .poster-info { padding:10px; } .poster-title { font-size:12px; } .poster-meta { font-size:10px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 4. SESSION STATE
# ============================================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "selected_title" not in st.session_state:
    st.session_state.selected_title = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "detail_return_page" not in st.session_state:
    st.session_state.detail_return_page = "Discover"

# ============================================================
# 5. HELPERS
# ============================================================
def clean_text(value, default="Not available"):
    """Safely convert missing or blank values to readable text."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return default
    result = str(value).strip()
    if not result or result.lower() in {"nan", "none", "nat"}:
        return default
    return result


def get_year(row):
    year = row.get("release_year", None)
    try:
        if pd.notna(year) and str(year).strip():
            return str(int(float(year)))
    except (TypeError, ValueError, OverflowError):
        pass
    return "Year unavailable"


def is_in_watchlist(show_id):
    return str(show_id) in {str(item) for item in st.session_state.watchlist}


def toggle_watchlist(show_id):
    show_id = str(show_id)
    if is_in_watchlist(show_id):
        st.session_state.watchlist = [
            str(item) for item in st.session_state.watchlist if str(item) != show_id
        ]
    else:
        st.session_state.watchlist.append(show_id)


def get_row_by_id(show_id):
    matches = df[df["show_id"].astype(str) == str(show_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_section_title(title, subtitle=""):
    st.markdown(f'<div class="section-title">{html.escape(str(title))}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{html.escape(str(subtitle))}</div>', unsafe_allow_html=True)


def render_metric(label, value, delta=None, help_text=None, accent=False):
    """Consistent metric helper used by the Home page."""
    st.metric(label=label, value=value, delta=delta, help=help_text)

# ============================================================
# 6. LOAD AND PREPARE DATA
# ============================================================
@st.cache_data

def load_data():
    if not DATA_PATH.exists():
        return None
    data = pd.read_csv(DATA_PATH)
    data.columns = [str(col).strip().lower().replace(" ", "_") for col in data.columns]
    required_columns = [
        "show_id", "type", "title", "director", "cast", "country",
        "date_added", "release_year", "rating", "duration", "listed_in", "description",
    ]
    for col in required_columns:
        if col not in data.columns:
            data[col] = np.nan
    for col in required_columns:
        data[col] = data[col].fillna("")
    data["title"] = data["title"].astype(str).replace({"nan": "", "None": ""})
    data["show_id"] = data["show_id"].astype(str)
    data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
    data["date_added"] = pd.to_datetime(data["date_added"], errors="coerce")
    data["genres"] = data["listed_in"].apply(
        lambda value: [genre.strip() for genre in str(value).split(",") if genre.strip()]
    )
    data["search_text"] = (
        data["title"].astype(str) + " " + data["director"].astype(str) + " "
        + data["cast"].astype(str) + " " + data["listed_in"].astype(str) + " "
        + data["description"].astype(str)
    )
    return data


df = load_data()
if df is None:
    st.error("Dataset not found. Make sure your CSV file is located at:")
    st.code("data/raw/netflix_titles.csv")
    st.info("Check the filename and folder structure in VS Code.")
    st.stop()

# ============================================================
# 7. TMDB POSTER API
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_poster(title, media_type):
    title = clean_text(title, "")
    if not title or title.lower() in {"nan", "none", "untitled"}:
        return ""
    media_type = str(media_type or "").strip().lower()
    endpoint_type = "tv" if media_type in {"tv show", "tv", "series"} else "movie"
    headers = {"Accept": "application/json"}
    if TMDB_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_ACCESS_TOKEN}"
        auth_params = {}
    elif TMDB_API_KEY:
        auth_params = {"api_key": TMDB_API_KEY}
    else:
        return ""

    def normalize(value):
        return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

    target = normalize(title)
    if not target:
        return ""
    simplified = re.sub(r"[:|/–—-]", " ", title)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    queries = [title]
    if simplified and simplified.lower() != title.lower():
        queries.append(simplified)
    base_url = f"https://api.themoviedb.org/3/search/{endpoint_type}"
    best_poster, best_score = "", 0.0
    try:
        for query in queries:
            response = requests.get(
                base_url,
                headers=headers,
                params={**auth_params, "query": query, "include_adult": "false", "page": 1, "language": "en-US"},
                timeout=8,
            )
            response.raise_for_status()
            for item in response.json().get("results", []):
                poster_path = item.get("poster_path")
                if not poster_path:
                    continue
                if endpoint_type == "tv":
                    candidate_names = [item.get("name", ""), item.get("original_name", "")]
                else:
                    candidate_names = [item.get("title", ""), item.get("original_title", "")]
                for candidate in candidate_names:
                    normalized_candidate = normalize(candidate)
                    if not normalized_candidate:
                        continue
                    score = 100.0 if normalized_candidate == target else SequenceMatcher(None, target, normalized_candidate).ratio() * 100
                    if score > best_score and score >= 85:
                        best_score, best_poster = score, poster_path
                if best_score == 100:
                    break
            if best_score == 100:
                break
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return ""
    return f"{TMDB_IMAGE_BASE}{best_poster}" if best_poster else ""

# ============================================================
# 8. MOVIE CARDS AND GRID
# ============================================================
def movie_card(row, key_prefix="card"):
    show_id = str(row.get("show_id", ""))
    title = clean_text(row.get("title", ""), "Untitled")
    content_type = clean_text(row.get("type", ""), "Movie")
    year = get_year(row)
    poster = get_tmdb_poster(title, content_type)
    genres = clean_text(row.get("listed_in", ""), "Genre unavailable")
    short_genres = genres[:55] + "..." if len(genres) > 58 else genres
    in_list = is_in_watchlist(show_id)
    watchlist_label = "✓ Added to My List" if in_list else "+ My List"
    safe_title = html.escape(title, quote=True)
    safe_type = html.escape(content_type, quote=True)
    safe_genres = html.escape(short_genres, quote=True)
    safe_poster = html.escape(poster, quote=True)
    safe_year = html.escape(year, quote=True)
    missing_class = " poster-missing" if not poster else ""
    image_style = "" if poster else ' style="display:none;"'
    st.markdown(
        f"""
        <div class="poster-card">
          <div class="poster-image-container{missing_class}">
            <img src="{safe_poster}" class="poster-image" alt="Poster for {safe_title}"
                 loading="lazy" decoding="async"{image_style}
                 onerror="this.style.display='none';this.parentElement.classList.add('poster-missing');" />
            <div class="poster-fallback"><span class="fallback-icon">🎬</span><span>POSTER UNAVAILABLE</span></div>
            <div class="poster-overlay"><span class="overlay-type">{safe_type}</span><span class="overlay-year">{safe_year}</span></div>
          </div>
          <div class="poster-info"><div class="poster-title" title="{safe_title}">{safe_title}</div>
          <div class="poster-meta">{safe_genres}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if st.button(watchlist_label, key=f"{key_prefix}_list_{show_id}", use_container_width=True):
            toggle_watchlist(show_id)
            st.rerun()
    with button_col2:
        if st.button("▶ Details", key=f"{key_prefix}_details_{show_id}", use_container_width=True, type="primary"):
            st.session_state.detail_return_page = st.session_state.get("navigation_page", "Discover")
            st.session_state.selected_title = show_id
            st.session_state.pending_page = "Title Details"
            st.rerun()


def render_grid(data, key_prefix="grid", columns=4):
    if data is None or data.empty:
        st.info("No titles match your selection.")
        return
    data = data.reset_index(drop=True)
    for start in range(0, len(data), columns):
        row_data = data.iloc[start:start + columns]
        cols = st.columns(columns)
        for index, (_, row) in enumerate(row_data.iterrows()):
            with cols[index]:
                movie_card(row, key_prefix=key_prefix)

# ============================================================
# 9. DETAILS PANEL
# ============================================================
def render_details():
    selected_id = st.session_state.selected_title
    if selected_id is None:
        return
    row = get_row_by_id(selected_id)
    if row is None:
        st.session_state.selected_title = None
        return
    st.markdown("---")
    render_section_title("Title Details", "Explore the selected title.")
    title = clean_text(row.get("title", ""), "Untitled")
    poster = get_tmdb_poster(title, clean_text(row.get("type", ""), "Movie"))
    with st.container(border=True):
        image_col, info_col = st.columns([1, 2], gap="large")
        with image_col:
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown('<div class="poster-fallback" style="position:relative;aspect-ratio:2/3;border-radius:12px;"><span class="fallback-icon">🎬</span><span>POSTER UNAVAILABLE</span></div>', unsafe_allow_html=True)
        with info_col:
            st.subheader(title)
            st.caption(f"{clean_text(row.get('type'))} • {get_year(row)} • {clean_text(row.get('rating'))}")
            st.markdown("**Description**")
            st.write(clean_text(row.get("description", ""), "No description available."))
            st.markdown("**Genres**")
            st.write(clean_text(row.get("listed_in", "")))
            st.markdown("**Director**")
            st.write(clean_text(row.get("director", "")))
            st.markdown("**Cast**")
            cast = clean_text(row.get("cast", ""))
            st.write(cast[:350] + ("..." if len(cast) > 350 else ""))
            st.markdown("**Country**")
            st.write(clean_text(row.get("country", "")))
            st.markdown("**Date Added**")
            added_date = row.get("date_added")
            st.write("Not available" if pd.isna(added_date) else added_date.strftime("%B %d, %Y"))
            st.markdown("**Duration**")
            st.write(clean_text(row.get("duration", "")))
            if st.button("✓ In My List" if is_in_watchlist(selected_id) else "+ Add to My List", key=f"details_watchlist_{selected_id}", type="primary"):
                toggle_watchlist(selected_id)
                st.rerun()
    if st.button("← Back to catalog", key="close_details", type="primary"):
        return_page = st.session_state.get("detail_return_page", "Discover")
        st.session_state.selected_title = None
        st.session_state.pending_page = return_page if return_page != "Title Details" else "Discover"
        st.rerun()

# ============================================================
# 10. CONTENT-BASED RECOMMENDATIONS
# ============================================================
@st.cache_resource
def build_recommendation_model(search_text_values):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = vectorizer.fit_transform(search_text_values)
    return matrix


def recommend_titles(show_id, limit=8):
    try:
        matrix = build_recommendation_model(tuple(df["search_text"].fillna("").astype(str)))
        matching_indices = df.index[df["show_id"].astype(str) == str(show_id)].tolist()
        if not matching_indices:
            return df.head(0)
        target_index = matching_indices[0]
        similarity_scores = cosine_similarity(matrix[target_index], matrix).flatten()
        similar_indices = np.argsort(similarity_scores)[::-1]
        results = []
        for index in similar_indices:
            if str(df.iloc[index]["show_id"]) == str(show_id):
                continue
            results.append(index)
            if len(results) >= limit:
                break
        return df.iloc[results]
    except (ValueError, TypeError):
        return df.head(0)

# ============================================================
# 11. SIDEBAR NAVIGATION
# ============================================================
if "pending_page" in st.session_state:
    st.session_state["navigation_page"] = st.session_state.pop("pending_page")
with st.sidebar:
    st.markdown('<div class="brand">NETFLIX</div><div class="brand-subtitle">INSIGHTS STUDIO</div>', unsafe_allow_html=True)
    st.markdown("### EXPLORE")
    nav_options = ["Home", "Discover", "Recommendations", "My List", "Analytics", "About"]
    if st.session_state.get("selected_title") is not None:
        nav_options.append("Title Details")
    if "navigation_page" not in st.session_state or st.session_state.navigation_page not in nav_options:
        st.session_state.navigation_page = "Home"
    page = st.radio("Navigate", nav_options, key="navigation_page", label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### YOUR COLLECTION")
    st.metric("Saved titles", len(st.session_state.watchlist))
    st.progress(min(len(st.session_state.watchlist) / 20, 1.0))
    st.caption("Your watchlist stays available during this session.")
    st.markdown("---")
    st.caption("DISCOVER · SAVE · ANALYZE")
    st.caption("Independent portfolio project, not affiliated with Netflix.")

# ============================================================
# 12. HOME PAGE
# ============================================================
if page == "Home":
    st.markdown("""<div class="hero"><div class="hero-tag">YOUR ENTERTAINMENT DATA HUB</div><h1>Discover Your Next Favorite.</h1><p>Explore the Netflix catalog through interactive discovery, personalized recommendations, and data-driven insights.</p></div>""", unsafe_allow_html=True)
    hero_cta1, hero_cta2 = st.columns([1, 1])
    with hero_cta1:
        if st.button("EXPLORE THE CATALOG →", key="hero_explore_catalog", type="primary", use_container_width=True):
            st.session_state.pending_page = "Discover"
            st.rerun()
    with hero_cta2:
        if st.button("YOUR NEXT FAVORITE STARTS HERE →", key="hero_next_favorite", use_container_width=True):
            st.session_state.pending_page = "Recommendations"
            st.rerun()
    action1, action2, action3 = st.columns(3)
    with action1:
        st.markdown("**Discover titles**")
        st.caption("Search and filter the catalog.")
        if st.button("Open Discover →", key="home_open_discover", use_container_width=True):
            st.session_state.pending_page = "Discover"; st.rerun()
    with action2:
        st.markdown("**Personal picks**")
        st.caption("Find titles similar to your favorites.")
        if st.button("Recommendations →", key="home_open_recommendations", use_container_width=True):
            st.session_state.pending_page = "Recommendations"; st.rerun()
    with action3:
        st.markdown("**Your collection**")
        st.caption("Return to titles you saved.")
        if st.button("Open My List →", key="home_open_mylist", use_container_width=True):
            st.session_state.pending_page = "My List"; st.rerun()
    total_titles = len(df)
    movie_count = int((df["type"].str.lower() == "movie").sum())
    tv_count = int((df["type"].str.lower() == "tv show").sum())
    genres_count = len({genre for genres in df["genres"] for genre in genres})
    metric_cols = st.columns(4)
    with metric_cols[0]: render_metric("TOTAL TITLES", f"{total_titles:,}", accent=True)
    with metric_cols[1]: render_metric("MOVIES", f"{movie_count:,}")
    with metric_cols[2]: render_metric("TV SHOWS", f"{tv_count:,}")
    with metric_cols[3]: render_metric("GENRES", f"{genres_count:,}")
    render_section_title("Recently Added", "Browse titles with the most recent catalog dates.")
    recent = df.dropna(subset=["date_added"]).sort_values("date_added", ascending=False).head(8)
    render_grid(recent, key_prefix="home_recent")
    render_section_title("Explore by Genre", "Choose a genre to discover matching titles.")
    all_genres = sorted({genre for genres in df["genres"] for genre in genres})
    if all_genres:
        selected_genre = st.selectbox("Choose your genre", all_genres, key="home_genre")
        genre_data = df[df["genres"].apply(lambda genres: selected_genre in genres)].head(8)
        render_grid(genre_data, key_prefix="home_genre")
    render_section_title("Your Watchlist", "Continue exploring the titles you've saved.")
    if st.session_state.watchlist:
        watchlist_data = df[df["show_id"].astype(str).isin([str(item) for item in st.session_state.watchlist])]
        render_grid(watchlist_data.head(4), key_prefix="home_watchlist")
    else:
        st.info("Your list is empty. Add titles from any movie card to build your personal collection.")
    if st.button("🎲 Surprise Me", type="primary", key="home_surprise"):
        random_row = df.sample(1).iloc[0]
        st.session_state.detail_return_page = "Home"
        st.session_state.selected_title = str(random_row["show_id"])
        st.session_state.pending_page = "Title Details"
        st.rerun()

# ============================================================
# 13. DISCOVER PAGE
# ============================================================
elif page == "Discover":
    st.title("Discover")
    st.caption("Search, filter, and explore the Netflix catalog.")
    search = st.text_input("🔍 Search titles, actors, directors, or descriptions", placeholder="Try Stranger Things, action, or a director...", key="discover_search")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        types = sorted(df["type"].replace("", np.nan).dropna().unique())
        selected_types = st.multiselect("Content Type", types, default=types, key="discover_types")
    with filter_col2:
        ratings = sorted(df["rating"].replace("", np.nan).dropna().unique())
        selected_ratings = st.multiselect("Rating", ratings, key="discover_ratings")
    with filter_col3:
        sort_by = st.selectbox("Sort By", ["Recently Added", "Newest Release", "Oldest Release", "Title A–Z", "Title Z–A"], key="discover_sort")
    filtered = df.copy()
    if search.strip():
        filtered = filtered[filtered["search_text"].str.contains(re.escape(search.strip()), case=False, na=False)]
    if selected_types:
        filtered = filtered[filtered["type"].isin(selected_types)]
    if selected_ratings:
        filtered = filtered[filtered["rating"].isin(selected_ratings)]
    if sort_by == "Recently Added": filtered = filtered.sort_values("date_added", ascending=False, na_position="last")
    elif sort_by == "Newest Release": filtered = filtered.sort_values("release_year", ascending=False, na_position="last")
    elif sort_by == "Oldest Release": filtered = filtered.sort_values("release_year", ascending=True, na_position="last")
    elif sort_by == "Title A–Z": filtered = filtered.sort_values("title")
    elif sort_by == "Title Z–A": filtered = filtered.sort_values("title", ascending=False)
    st.markdown("---")
    result_col1, result_col2 = st.columns(2)
    with result_col1: st.metric("Matching Titles", f"{len(filtered):,}")
    with result_col2: st.metric("Saved Titles", len(st.session_state.watchlist))
    page_size = 16
    total_pages = max(1, int(np.ceil(len(filtered) / page_size)))
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="discover_page")
    page_data = filtered.iloc[(current_page - 1) * page_size: (current_page - 1) * page_size + page_size]
    st.caption(f"Page {current_page} of {total_pages}")
    render_grid(page_data, key_prefix=f"discover_{current_page}")

# ============================================================
# 14. RECOMMENDATIONS PAGE
# ============================================================
elif page == "Recommendations":
    st.title("Personalized Recommendations")
    st.caption("Discover titles related to your favorite content using content-based machine learning.")
    title_options = df[["show_id", "title", "type", "release_year"]].copy()
    title_options["display"] = title_options.apply(lambda r: f"{clean_text(r['title'], 'Untitled')} ({clean_text(r['type'], 'Unknown')} - {get_year(r)})", axis=1)
    selected_display = st.selectbox("Choose a title you like", title_options["display"].tolist(), key="recommendation_title")
    if st.button("✨ Find Similar Titles", type="primary", key="recommendation_button"):
        selected_row = title_options[title_options["display"] == selected_display].iloc[0]
        st.session_state.recommendation_source = str(selected_row["show_id"])
    if "recommendation_source" in st.session_state:
        source_id = st.session_state.recommendation_source
        source_row = get_row_by_id(source_id)
        if source_row is not None:
            st.markdown("---")
            st.subheader(f"Because you selected: {clean_text(source_row['title'], 'Untitled')}")
            render_grid(recommend_titles(source_id, limit=8), key_prefix="recommendations")

# ============================================================
# 15. MY LIST PAGE
# ============================================================
elif page == "My List":
    st.markdown("""<div class="hero" style="min-height:210px;padding:32px;"><div class="hero-tag">YOUR PERSONAL COLLECTION</div><h1 style="font-size:clamp(30px,4vw,46px);">My List</h1><p>Everything you saved, gathered in one place. Keep track of what you want to watch next.</p></div>""", unsafe_allow_html=True)
    watchlist_ids = [str(item) for item in st.session_state.watchlist]
    watchlist_data = df[df["show_id"].astype(str).isin(watchlist_ids)]
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1: st.metric("Titles Saved", len(watchlist_data))
    with header_col2:
        if not watchlist_data.empty and st.button("🗑 Clear My List", key="clear_watchlist"):
            st.session_state.confirm_clear_watchlist = True
    if st.session_state.get("confirm_clear_watchlist", False):
        st.warning("This will remove every saved title from your list.")
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("Confirm clear", type="primary", key="confirm_clear_watchlist_yes"):
                st.session_state.watchlist = []
                st.session_state.selected_title = None
                st.session_state.confirm_clear_watchlist = False
                st.rerun()
        with confirm_col2:
            if st.button("Cancel", key="confirm_clear_watchlist_no"):
                st.session_state.confirm_clear_watchlist = False
                st.rerun()
    if not watchlist_data.empty:
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1: list_query = st.text_input("Search your list", placeholder="Search saved titles...", key="mylist_search")
        with filter_col2: list_type = st.selectbox("Content type", ["All", "Movie", "TV Show"], key="mylist_type")
        if list_query.strip():
            watchlist_data = watchlist_data[watchlist_data["title"].str.contains(re.escape(list_query.strip()), case=False, na=False)]
        if list_type != "All":
            watchlist_data = watchlist_data[watchlist_data["type"].str.lower() == list_type.lower()]
        st.caption(f"Showing {len(watchlist_data)} saved title(s).")
    if watchlist_data.empty:
        st.markdown('''<div class="empty-state"><h2>🎬 Your story starts here.</h2><p>You haven't saved any titles yet. Explore the catalog and add movies or shows that catch your attention.</p></div>''', unsafe_allow_html=True)
        if st.button("Explore the Catalog", type="primary", key="mylist_explore"):
            st.session_state.pending_page = "Discover"
            st.rerun()
    else:
        render_grid(watchlist_data, key_prefix="mylist")

# ============================================================
# 16. ANALYTICS DASHBOARD
# ============================================================
elif page == "Analytics":
    st.markdown("""<div class="hero" style="min-height:220px;padding:34px;"><div class="hero-tag">CATALOG INTELLIGENCE</div><h1 style="font-size:clamp(30px,4vw,46px);">Analytics Studio</h1><p>Explore catalog composition, release patterns, genres, geography, maturity ratings, and Netflix additions. Use the controls to focus every insight on the titles you care about.</p></div>""", unsafe_allow_html=True)

    # Interactive analytics toolbar
    st.markdown('<div class="analytics-toolbar"><h3 style="margin-top:0;">🎛️ Analysis Controls</h3><p style="color:#aaa;font-size:13px;margin-bottom:10px;">All charts and KPIs respond to these filters.</p></div>', unsafe_allow_html=True)
    filter_row1 = st.columns(3)
    with filter_row1[0]:
        analytics_type = st.selectbox("Content type", ["All titles", "Movies", "TV Shows"], key="analytics_content_type")
    years_available = sorted(int(y) for y in df["release_year"].dropna().unique())
    with filter_row1[1]:
        year_range = st.slider("Release year range", min_value=min(years_available), max_value=max(years_available), value=(min(years_available), max(years_available)), key="analytics_year_range") if years_available else None
    with filter_row1[2]:
        top_n = st.select_slider("Top categories to show", options=[5, 8, 10, 12, 15, 20], value=10, key="analytics_top_n")

    with st.expander("More filters", expanded=True):
        filter_row2 = st.columns(3)
        all_ratings = sorted(x for x in df["rating"].dropna().astype(str).unique() if x.strip())
        all_genres = sorted({g for values in df["genres"] for g in values if g.strip()})
        all_countries = sorted({country.strip() for value in df["country"].fillna("").astype(str) for country in value.split(",") if country.strip() and country.strip().lower() != "nan"})
        with filter_row2[0]: selected_ratings = st.multiselect("Maturity rating", all_ratings, key="analytics_ratings")
        with filter_row2[1]: selected_genres = st.multiselect("Genres", all_genres, key="analytics_genres")
        with filter_row2[2]: selected_countries = st.multiselect("Country / production region", all_countries, key="analytics_countries")
        date_years = sorted(int(y) for y in df["date_added"].dropna().dt.year.unique())
        filter_row3 = st.columns([1, 1, 1])
        with filter_row3[0]:
            added_year_range = st.select_slider("Netflix date-added years", options=date_years, value=(date_years[0], date_years[-1]), key="analytics_added_years") if date_years else None
        with filter_row3[1]:
            chart_style = st.selectbox("Trend chart style", ["Area", "Line", "Bar"], key="analytics_trend_style")
        with filter_row3[2]:
            st.caption("Tip: select multiple genres or countries to compare a focused subset.")

    analytics_df = df.copy()
    if analytics_type == "Movies": analytics_df = analytics_df[analytics_df["type"].str.lower() == "movie"]
    elif analytics_type == "TV Shows": analytics_df = analytics_df[analytics_df["type"].str.lower() == "tv show"]
    if year_range:
        analytics_df = analytics_df[analytics_df["release_year"].isna() | analytics_df["release_year"].between(year_range[0], year_range[1])]
    if selected_ratings:
        analytics_df = analytics_df[analytics_df["rating"].astype(str).isin(selected_ratings)]
    if selected_genres:
        analytics_df = analytics_df[analytics_df["genres"].apply(lambda values: any(g in values for g in selected_genres))]
    if selected_countries:
        analytics_df = analytics_df[analytics_df["country"].fillna("").astype(str).apply(lambda value: any(c in [x.strip() for x in value.split(",")] for c in selected_countries))]
    if added_year_range:
        added_years = analytics_df["date_added"].dt.year
        analytics_df = analytics_df[added_years.isna() | added_years.between(added_year_range[0], added_year_range[1])]

    st.caption(f"Showing **{len(analytics_df):,}** titles after the selected filters. Charts below are interactive: hover for values and use the chart toolbar to zoom, pan, or download.")
    total_titles = len(analytics_df)
    movie_count = int((analytics_df["type"].str.lower() == "movie").sum())
    show_count = int((analytics_df["type"].str.lower() == "tv show").sum())
    valid_years = analytics_df["release_year"].dropna()
    latest_year = int(valid_years.max()) if not valid_years.empty else "N/A"
    movie_share = movie_count / total_titles * 100 if total_titles else 0
    avg_release_year = int(round(valid_years.mean())) if not valid_years.empty else "N/A"
    added_dates = analytics_df["date_added"].dropna()
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1: st.metric("Titles in View", f"{total_titles:,}")
    with kpi2: st.metric("Movies", f"{movie_count:,}", f"{movie_share:.1f}% of selection" if total_titles else None)
    with kpi3: st.metric("TV Shows", f"{show_count:,}")
    with kpi4: st.metric("Latest Release", latest_year)
    with kpi5: st.metric("Median Release Year", int(valid_years.median()) if not valid_years.empty else "N/A")

    def chart_layout(fig, x_title=None, y_title=None, height=360):
        fig.update_layout(template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#101010", font_color="white", margin=dict(l=15, r=15, t=35, b=15), height=height, legend_title_text="")
        if x_title is not None: fig.update_xaxes(title=x_title, gridcolor="#292929")
        if y_title is not None: fig.update_yaxes(title=y_title, gridcolor="#292929")
        return fig

    def show_insight(message):
        st.markdown(f'<div class="insight-box"><strong>💡 What this tells you</strong><br>{html.escape(str(message))}</div>', unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Content Type Mix")
        type_counts = analytics_df["type"].replace("", np.nan).dropna().value_counts().rename_axis("Content Type").reset_index(name="Titles")
        if not type_counts.empty:
            fig = px.pie(type_counts, names="Content Type", values="Titles", hole=0.62, color_discrete_sequence=["#e50914", "#555555"], custom_data=["Titles"])
            fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>Titles: %{value:,}<br>Share: %{percent}<extra></extra>")
            chart_layout(fig, height=360)
            st.plotly_chart(fig, use_container_width=True, key="analytics_type_chart")
            leading = type_counts.iloc[0]
            share = leading["Titles"] / max(total_titles, 1) * 100
            show_insight(f"{leading['Content Type']} is the larger content group with {int(leading['Titles']):,} titles ({share:.1f}% of the current selection). The split changes as you apply content, year, rating, genre, or country filters.")
        else: st.info("No content-type data matches these filters.")

    with right:
        st.subheader("Titles by Release Year")
        releases = analytics_df.dropna(subset=["release_year"]).groupby("release_year").size().reset_index(name="Titles").sort_values("release_year")
        if not releases.empty:
            if chart_style == "Line": fig = px.line(releases, x="release_year", y="Titles", markers=True, color_discrete_sequence=["#e50914"])
            elif chart_style == "Bar": fig = px.bar(releases, x="release_year", y="Titles", color_discrete_sequence=["#e50914"])
            else: fig = px.area(releases, x="release_year", y="Titles", color_discrete_sequence=["#e50914"])
            chart_layout(fig, "Original release year", "Number of catalog titles", 360)
            st.plotly_chart(fig, use_container_width=True, key="analytics_release_chart")
            peak = releases.loc[releases["Titles"].idxmax()]
            show_insight(f"The largest release-year cohort in this selection is {int(peak['release_year'])}, with {int(peak['Titles']):,} titles. This chart shows when titles were originally released, not when they were added to Netflix.")
        else: st.info("No release-year data matches these filters.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Most Common Genres")
        genre_data = analytics_df.explode("genres").copy()
        genre_data["genres"] = genre_data["genres"].astype(str).str.strip()
        genre_data = genre_data[genre_data["genres"].ne("") & genre_data["genres"].str.lower().ne("nan")].groupby("genres").size().reset_index(name="Titles")
        genre_data = genre_data.sort_values("Titles", ascending=False).head(top_n).sort_values("Titles")
        if not genre_data.empty:
            fig = px.bar(genre_data, x="Titles", y="genres", orientation="h", color="Titles", color_continuous_scale=["#4b1115", "#e50914"], hover_data={"Titles": ":,"})
            chart_layout(fig, "Titles tagged with genre", "", max(360, 28 * len(genre_data)))
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key="analytics_genres_chart")
            show_insight(f"{genre_data.iloc[-1]['genres']} is the most frequent genre in the displayed top {len(genre_data)}. A title can belong to several genres, so genre counts overlap and should not be added together as unique titles.")
        else: st.info("No genre data matches these filters.")

    with right:
        st.subheader("Top Production Countries")
        country_data = analytics_df.assign(country=analytics_df["country"].fillna("").astype(str).str.split(",")).explode("country")
        country_data["country"] = country_data["country"].astype(str).str.strip()
        country_counts = country_data[country_data["country"].ne("") & country_data["country"].str.lower().ne("nan")].groupby("country").size().reset_index(name="Titles").sort_values("Titles", ascending=False).head(top_n).sort_values("Titles")
        if not country_counts.empty:
            fig = px.bar(country_counts, x="Titles", y="country", orientation="h", color="Titles", color_continuous_scale=["#333333", "#e50914"], hover_data={"Titles": ":,"})
            chart_layout(fig, "Titles associated with country", "", max(360, 28 * len(country_counts)))
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key="analytics_countries_chart")
            show_insight(f"{country_counts.iloc[-1]['country']} is the most represented country in the displayed ranking. International co-productions can appear under multiple countries, so these counts are title-country associations rather than exclusive shares.")
        else: st.info("No country data matches these filters.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Maturity Rating Profile")
        rating_data = analytics_df[analytics_df["rating"].astype(str).str.strip().ne("")].groupby("rating").size().reset_index(name="Titles").sort_values("Titles", ascending=False).head(top_n).sort_values("Titles")
        if not rating_data.empty:
            fig = px.bar(rating_data, x="Titles", y="rating", orientation="h", color="Titles", color_continuous_scale=["#333333", "#e50914"])
            chart_layout(fig, "Titles", "Maturity rating", max(330, 27 * len(rating_data)))
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key="analytics_ratings_chart")
            show_insight(f"The most common maturity rating among the displayed categories is {rating_data.iloc[-1]['rating']}. Ratings describe audience guidance, not a quality score; missing ratings are not included in this chart.")
        else: st.info("No rating data matches these filters.")

    with right:
        st.subheader("Titles Added to Netflix Over Time")
        added = analytics_df.dropna(subset=["date_added"]).assign(AddedYear=lambda x: x["date_added"].dt.year).groupby("AddedYear").size().reset_index(name="Titles").sort_values("AddedYear")
        if not added.empty:
            if chart_style == "Line": fig = px.line(added, x="AddedYear", y="Titles", markers=True, color_discrete_sequence=["#e50914"])
            elif chart_style == "Bar": fig = px.bar(added, x="AddedYear", y="Titles", color_discrete_sequence=["#e50914"])
            else: fig = px.area(added, x="AddedYear", y="Titles", color_discrete_sequence=["#e50914"])
            chart_layout(fig, "Year added to Netflix", "Number of titles added", 330)
            st.plotly_chart(fig, use_container_width=True, key="analytics_added_chart")
            peak = added.loc[added["Titles"].idxmax()]
            show_insight(f"The largest recorded addition year in this filtered dataset is {int(peak['AddedYear'])}, with {int(peak['Titles']):,} titles. This is the year shown in the dataset's date_added field and may not represent every title ever available on Netflix.")
        else: st.info("No date-added data matches these filters.")

    # Duration analysis where source values can be interpreted numerically
    st.subheader("Duration Snapshot")
    duration_df = analytics_df.copy()
    duration_df["duration_value"] = pd.to_numeric(duration_df["duration"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    movie_durations = duration_df[(duration_df["type"].str.lower() == "movie") & duration_df["duration_value"].notna()]
    tv_durations = duration_df[(duration_df["type"].str.lower() == "tv show") & duration_df["duration_value"].notna()]
    duration_col1, duration_col2, duration_col3 = st.columns(3)
    with duration_col1: st.metric("Movies with duration data", f"{len(movie_durations):,}")
    with duration_col2: st.metric("Median movie length", f"{movie_durations['duration_value'].median():.0f} min" if not movie_durations.empty else "N/A")
    with duration_col3: st.metric("Median TV seasons", f"{tv_durations['duration_value'].median():.0f}" if not tv_durations.empty else "N/A")
    if not movie_durations.empty:
        duration_hist = px.histogram(movie_durations, x="duration_value", nbins=20, color_discrete_sequence=["#e50914"], labels={"duration_value":"Movie duration (minutes)"})
        chart_layout(duration_hist, "Movie runtime (minutes)", "Number of movies", 300)
        st.plotly_chart(duration_hist, use_container_width=True, key="analytics_duration_hist")
        show_insight("This histogram shows the distribution of movie runtimes in minutes. The median above is less affected by unusually short or long titles than the average. TV duration values are seasons, so they are summarized separately and are not mixed into the movie-runtime chart.")
    else:
        st.info("No movie-duration values are available for the selected filters.")

    # Executive summary / conclusion generated from current filters
    st.markdown("---")
    st.subheader("📌 Summary of the Current Selection")
    if total_titles:
        top_genre_name = "unavailable"
        if not analytics_df.explode("genres").empty:
            summary_genres = analytics_df.explode("genres").groupby("genres").size().sort_values(ascending=False)
            summary_genres = summary_genres[
    summary_genres.index.astype(str).str.strip() != ""
]
            if not summary_genres.empty: top_genre_name = str(summary_genres.index[0])
        top_country_name = "unavailable"
        summary_countries = analytics_df.assign(country=analytics_df["country"].fillna("").astype(str).str.split(",")).explode("country")
        summary_countries["country"] = summary_countries["country"].astype(str).str.strip()
        summary_countries = summary_countries[summary_countries["country"].ne("") & summary_countries["country"].str.lower().ne("nan")].groupby("country").size().sort_values(ascending=False)
        if not summary_countries.empty: top_country_name = str(summary_countries.index[0])
        st.markdown(f'<div class="insight-box"><strong>In plain English</strong><br>The current filters leave <b>{total_titles:,} titles</b>: {movie_count:,} movies and {show_count:,} TV shows. Movies represent {movie_share:.1f}% of this selection. The most frequent genre is <b>{html.escape(top_genre_name)}</b>, and the most represented production country is <b>{html.escape(top_country_name)}</b>. Use the filters above to compare different slices of the catalog. Remember that genres and countries can overlap for a single title.</div>', unsafe_allow_html=True)
    else:
        st.warning("No titles match the current filter combination. Clear one or more filters to see insights again.")

    st.markdown("---")
    st.subheader("Explore and Export the Filtered Dataset")
    st.write("Preview the exact rows used by the current analysis and download them for further exploration in Excel, Power BI, or Python.")
    clean_export = analytics_df.drop(columns=["genres", "search_text"], errors="ignore")
    st.dataframe(clean_export, use_container_width=True, height=350)
    st.download_button("⬇ Download Filtered Dataset as CSV", data=clean_export.to_csv(index=False).encode("utf-8"), file_name="netflix_insights_filtered_dataset.csv", mime="text/csv", type="primary")

# ============================================================
# 17. TITLE DETAILS PAGE
# ============================================================
elif page == "Title Details":
    st.markdown("""<div class="hero" style="min-height:180px;padding:28px;"><div class="hero-tag">TITLE PROFILE</div><h1 style="font-size:clamp(30px,4vw,44px);">Details & Credits</h1><p>Explore the title information, description, cast, and catalog metadata.</p></div>""", unsafe_allow_html=True)
    render_details()

# ============================================================
# 18. ABOUT PAGE
# ============================================================
elif page == "About":
    st.title("About Netflix Insights")
    st.markdown("""<div class="hero"><div class="hero-tag">DATA SCIENCE PORTFOLIO</div><h1>Entertainment Meets Data.</h1><p>Netflix Insights is an interactive application designed to explore a streaming catalog using data analysis, visualization, and machine learning.</p></div>""", unsafe_allow_html=True)
    st.subheader("Project Overview")
    st.write("This project transforms a raw entertainment dataset into an interactive discovery and analytics experience.")
    about_col1, about_col2 = st.columns(2)
    with about_col1:
        st.markdown("### Technologies")
        st.markdown("- **Python** — application logic\n- **Pandas** — data cleaning and manipulation\n- **NumPy** — numerical operations\n- **Scikit-learn** — recommendation engine\n- **Plotly** — interactive charts\n- **Streamlit** — application interface\n- **TMDB API** — movie and TV posters")
    with about_col2:
        st.markdown("### Key Features")
        st.markdown("- Search and filter catalog titles\n- Interactive movie and TV details\n- Personal watchlist\n- Content-based recommendations\n- Genre and country analysis\n- Release-year trends\n- Downloadable dataset")
    st.markdown("---")
    st.subheader("Data Source")
    st.write("This project uses the Netflix titles dataset provided in the project's data folder.")
    st.caption("Netflix Insights is an independent portfolio project. It is not affiliated with Netflix.")
    st.caption("Movie and TV poster images are requested from The Movie Database (TMDB).")

# ============================================================
# 19. FOOTER
# ============================================================
st.markdown("""<div class="footer">NETFLIX INSIGHTS STUDIO<br><br>Built with Python, Streamlit, and data science.<br>Independent portfolio project. Not affiliated with Netflix.</div>""", unsafe_allow_html=True)
