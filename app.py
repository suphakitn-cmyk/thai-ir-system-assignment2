import streamlit as st
import math
import re
from collections import defaultdict, Counter
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Thai IR System - Assignment 2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 2. CUSTOM DESIGN SYSTEM (FULL STYLES)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif !important;
        background-color: #F8FAFC !important;
    }

    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global App Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Top Banner / Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 32px 40px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 8px;
        font-weight: 300;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        padding: 18px 22px !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #64748B !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Tab Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        white-space: pre;
        background-color: #FFFFFF;
        border-radius: 10px 10px 0px 0px;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        color: #64748B;
        font-weight: 500;
        padding: 0px 22px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
        font-weight: 600 !important;
    }

    /* Document Cards Specs */
    .doc-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: all 0.25s ease-in-out;
    }
    .doc-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 8px 18px -4px rgba(59, 130, 246, 0.12);
        transform: translateY(-2px);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .doc-badge {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .score-badge {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .doc-text {
        font-size: 1.05rem;
        color: #334155;
        line-height: 1.6;
        margin: 0;
    }

    /* Search Box Input Field Styling */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        padding: 4px 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. DATA & LOGIC ENGINE
# ---------------------------------------------------------
documents = [
    "สวัสดีครับ นี่คือข้อความทดสอบสำหรับการประมวลผลข้อความ",
    "เราจะตัดคำ ลดรูปคำ และลบคำฟุ่มเฟือยออกจากข้อความนี้",
    "จากนั้นสร้างดัชนีผกผันเพื่อค้นหาข้อมูลได้อย่างมีประสิทธิภาพ"
]

@st.cache_data
def get_stopwords():
    return set(thai_stopwords())

stopwords = get_stopwords()

def preprocess_text(text):
    tokens = word_tokenize(text, engine="newmm")
    return [
        token.strip() for token in tokens 
        if token.strip() != "" and token not in stopwords and not re.match(r'^\s+$', token)
    ]

doc_tokens = {i+1: preprocess_text(doc) for i, doc in enumerate(documents)}

def compute_tf(tokens):
    tf_dict = {}
    total_tokens = len(tokens)
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        tf_dict[token] = count / total_tokens if total_tokens > 0 else 0
    return tf_dict

def compute_idf(doc_tokens_dict):
    N = len(doc_tokens_dict)
    idf_dict = {}
    all_words = set(word for tokens in doc_tokens_dict.values() for word in tokens)
    for word in all_words:
        doc_count = sum(1 for tokens in doc_tokens_dict.values() if word in tokens)
        idf_dict[word] = math.log10(N / doc_count) if doc_count > 0 else 0
    return idf_dict

def compute_tfidf(doc_tokens_dict):
    idf_dict = compute_idf(doc_tokens_dict)
    tfidf_dict = {}
    for doc_id, tokens in doc_tokens_dict.items():
        tf = compute_tf(tokens)
        tfidf_dict[doc_id] = {word: tf[word] * idf_dict[word] for word in tf}
    return tfidf_dict, idf_dict

tfidf_results, idf_results = compute_tfidf(doc_tokens)

# ---------------------------------------------------------
# 4. UI RENDERING
# ---------------------------------------------------------

# HERO HEADER
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Thai IR System with Term Weighting</div>
    <div class="hero-subtitle">ระบบค้นคืนสารสนเทศภาษาไทย ประมวลผลและจัดอันดับความเกี่ยวข้องด้วย TF-IDF Model (Assignment 2)</div>
</div>
""", unsafe_allow_html=True)

# METRIC DASHBOARD
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric(label="📚 จำนวนเอกสารในระบบ (Corpus)", value=f"{len(documents)} รายการ")
with col_m2:
    st.metric(label="🔤 จำนวนคำศัพท์ทั้งหมด (Vocabulary)", value=f"{len(idf_results)} คำ")
with col_m3:
    st.metric(label="⚙️ Algorithm Engine", value="TF-IDF Weighting")

st.markdown("<br>", unsafe_allow_html=True)

# TABBED NAVIGATION
tab_search, tab_matrix, tab_docs = st.tabs([
    "🔍 ค้นหาข้อมูล (Search Engine)", 
    "📊 ตารางน้ำหนักคำ (TF-IDF Matrix)", 
    "📄 คลังเอกสาร (Corpus View)"
])

# --- TAB 1: SEARCH ---
with tab_search:
    st.markdown("<br>", unsafe_allow_html=True)
    query = st.text_input("ป้อนคำค้นหาที่ต้องการ (Query):", placeholder="เช่น ข้อความ, ค้นหา, ดัชนี")
    
    if query:
        query_tokens = preprocess_text(query)
        st.markdown(f"**⚙️ คำค้นหาหลัง Preprocessing:** `{' | '.join(query_tokens) if query_tokens else 'ไม่พบคำสำคัญ'}`")
        st.markdown("<br>", unsafe_allow_html=True)
        
        scores = defaultdict(float)
        for token in query_tokens:
            for doc_id, tfidf_map in tfidf_results.items():
                if token in tfidf_map:
                    scores[doc_id] += tfidf_map[token]
                    
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        st.markdown("### 🎯 ผลการค้นหา (Ranked Results)")
        
        found = False
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            if score > 0:
                found = True
                st.markdown(f"""
                <div class="doc-card">
                    <div class="card-header">
                        <span class="doc-badge">อันดับที่ {rank} • Document {doc_id}</span>
                        <span class="score-badge">TF-IDF Score: {score:.4f}</span>
                    </div>
                    <p class="doc-text">{documents[doc_id-1]}</p>
                </div>
                """, unsafe_allow_html=True)
                
        if not found:
            st.warning("❌ ไม่พบเอกสารที่เกี่ยวข้องกับคำค้นหานี้")

# --- TAB 2: TF-IDF MATRIX ---
with tab_matrix:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 ตารางแสดงค่าน้ำหนัก Term Weighting (TF-IDF)")
    st.caption("ตารางแสดงค่า Inverse Document Frequency (IDF) ของคำศัพท์ และค่า TF-IDF ในแต่ละเอกสาร")
    
    all_words = sorted(list(idf_results.keys()))
    matrix = []
    for word in all_words:
        row = {'คีย์เวิร์ด (Term)': word, 'IDF Score': round(idf_results[word], 4)}
        for doc_id in doc_tokens.keys():
            row[f'Doc {doc_id} (TF-IDF)'] = round(tfidf_results[doc_id].get(word, 0.0), 4)
        matrix.append(row)
        
    st.dataframe(
        pd.DataFrame(matrix), 
        use_container_width=True, 
        height=420
    )

# --- TAB 3: CORPUS ---
with tab_docs:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📄 เอกสารทั้งหมดในคลังข้อมูล (Corpus)")
    st.caption("รายการเอกสารต้นฉบับที่ระบบนำมาประมวลผลดัชนี")
    
    for doc_id, doc_text in enumerate(documents, start=1):
        st.markdown(f"""
        <div class="doc-card">
            <div class="card-header">
                <span class="doc-badge">Document {doc_id}</span>
            </div>
            <p class="doc-text">{doc_text}</p>
        </div>
        """, unsafe_allow_html=True)
