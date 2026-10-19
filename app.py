import streamlit as st
import math
import re
from collections import defaultdict, Counter
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# 1. ตั้งค่าหน้าเว็บแบบกว้าง พร้อมใส่ Icon และชื่อ
st.set_page_config(
    page_title="Thai IR System - Assignment 2",
    page_icon="⚡",
    layout="wide"
)

# 2. ปรับแต่งดีไซน์ด้วย Custom CSS (Modern Light Theme)
st.markdown("""
    <style>
    /* ซ่อน Streamlit Header & Footer ส่วนเกิน */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ตกแต่ง Header ใหญ่ */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* ตกแต่งการ์ดแสดงเอกสาร */
    .doc-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .doc-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .doc-badge {
        background-color: #3B82F6;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .score-badge {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        float: right;
    }
    .doc-text {
        font-size: 1.05rem;
        color: #334155;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# --- ข้อมูลและฟังก์ชันประมวลผล ---
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

# --- ส่วนการแสดงผลส่วนหัว (Header) ---
st.markdown('<div class="main-title">⚡ Thai IR System with Term Weighting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ระบบค้นคืนสารสนเทศภาษาไทย ประมวลผลและจัดอันดับความเกี่ยวข้องด้วย TF-IDF (Assignment 2)</div>', unsafe_allow_html=True)

# การ์ดสรุปสถิติระบบ (Metrics Summary)
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric(label="📚 จำนวนเอกสารในระบบ", value=f"{len(documents)} รายการ")
with col_m2:
    st.metric(label="🔤 คำศัพท์ทั้งหมด (Vocabulary)", value=f"{len(idf_results)} คำ")
with col_m3:
    st.metric(label="⚙️ Algorithm", value="TF-IDF Weighting")

st.divider()

# --- แยกหน้าการทำงานเป็น Tabs เพื่อ UX ที่สะอาดขึ้น ---
tab_search, tab_matrix, tab_docs = st.tabs(["🔍 ค้นหาข้อมูล (Search)", "📊 ตารางน้ำหนักคำ (TF-IDF Table)", "📄 เอกสารทั้งหมด (Corpus)"])

# TAB 1: หน้าค้นหาข้อมูล
with tab_search:
    query = st.text_input("พิมพ์คำค้นหาที่ต้องการ:", placeholder="เช่น ข้อความ, ค้นหา, ดัชนี")
    
    if query:
        query_tokens = preprocess_text(query)
        st.caption(f"⚙️ **คำค้นหาหลังทำ Preprocessing:** `{ ' | '.join(query_tokens) if query_tokens else 'ไม่พบคำสำคัญ' }`")
        
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
                    <span class="doc-badge">อันดับที่ {rank} • Document {doc_id}</span>
                    <span class="score-badge">TF-IDF Score: {score:.4f}</span>
                    <div class="doc-text">{documents[doc_id-1]}</div>
                </div>
                """, unsafe_allow_html=True)
                
        if not found:
            st.info("❌ ไม่พบเอกสารที่เกี่ยวข้องกับคำค้นหานี้")

# TAB 2: ตารางแสดงน้ำหนักคำ
with tab_matrix:
    st.markdown("#### 📊 ค่าน้ำหนัก Term Weighting (TF-IDF)")
    st.write("ตารางแสดงค่า IDF ของแต่ละคำ และค่า TF-IDF ในแต่ละเอกสาร")
    
    all_words = sorted(list(idf_results.keys()))
    matrix = []
    for word in all_words:
        row = {'คีย์เวิร์ด (Term)': word, 'IDF': round(idf_results[word], 4)}
        for doc_id in doc_tokens.keys():
            row[f'Doc {doc_id} (TF-IDF)'] = round(tfidf_results[doc_id].get(word, 0.0), 4)
        matrix.append(row)
        
    st.dataframe(pd.DataFrame(matrix), use_container_width=True, height=400)

# TAB 3: แสดงรายการเอกสารทั้งหมด
with tab_docs:
    st.markdown("#### 📄 รายการเอกสารในคลังข้อมูล")
    for doc_id, doc_text in enumerate(documents, start=1):
        st.markdown(f"""
        <div class="doc-card">
            <span class="doc-badge">Document {doc_id}</span>
            <div class="doc-text">{doc_text}</div>
        </div>
        """, unsafe_allow_html=True)
