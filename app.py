import streamlit as st
import math
import re
from collections import defaultdict, Counter
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Thai IR System with TF-IDF", page_icon="📊", layout="wide")

st.title("📊 Thai IR System - Term Weighting (Assignment 2)")
st.write("ระบบค้นคืนสารสนเทศภาษาไทยที่เพิ่มประสิทธิภาพการให้น้ำหนักคำด้วย TF-IDF")

# เอกสารตัวอย่าง
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

# 1. คำนวณ Term Frequency (TF)
def compute_tf(tokens):
    tf_dict = {}
    total_tokens = len(tokens)
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        tf_dict[token] = count / total_tokens if total_tokens > 0 else 0
    return tf_dict

# 2. คำนวณ Inverse Document Frequency (IDF)
def compute_idf(doc_tokens_dict):
    N = len(doc_tokens_dict)
    idf_dict = {}
    all_words = set(word for tokens in doc_tokens_dict.values() for word in tokens)
    for word in all_words:
        doc_count = sum(1 for tokens in doc_tokens_dict.values() if word in tokens)
        idf_dict[word] = math.log10(N / doc_count) if doc_count > 0 else 0
    return idf_dict

# 3. คำนวณ TF-IDF
def compute_tfidf(doc_tokens_dict):
    idf_dict = compute_idf(doc_tokens_dict)
    tfidf_dict = {}
    for doc_id, tokens in doc_tokens_dict.items():
        tf = compute_tf(tokens)
        tfidf_dict[doc_id] = {word: tf[word] * idf_dict[word] for word in tf}
    return tfidf_dict, idf_dict

tfidf_results, idf_results = compute_tfidf(doc_tokens)

# ตารางแสดงค่าน้ำหนัก TF-IDF
with st.expander("🔍 ดูตารางค่าน้ำหนักคีย์เวิร์ด (Term Weighting Table)"):
    all_words = sorted(list(idf_results.keys()))
    matrix = []
    for word in all_words:
        row = {'คีย์เวิร์ด (Term)': word, 'IDF': round(idf_results[word], 4)}
        for doc_id in doc_tokens.keys():
            row[f'Doc {doc_id} (TF-IDF)'] = round(tfidf_results[doc_id].get(word, 0.0), 4)
        matrix.append(row)
    st.dataframe(pd.DataFrame(matrix), use_container_width=True)

# ช่องพิมพ์คำค้นหา
query = st.text_input("ใส่คำค้นหาที่ต้องการ (Query):", placeholder="เช่น ข้อความ, ค้นหา")

if query:
    query_tokens = preprocess_text(query)
    st.write(f"**คำค้นหาหลัง Preprocessing:** `{query_tokens}`")
    
    scores = defaultdict(float)
    for token in query_tokens:
        for doc_id, tfidf_map in tfidf_results.items():
            if token in tfidf_map:
                scores[doc_id] += tfidf_map[token]
                
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    st.subheader("🎯 ผลการค้นหา (เรียงตามคะแนน TF-IDF)")
    if ranked_docs and ranked_docs[0][1] > 0:
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            if score > 0:
                st.success(f"**อันดับ {rank}** | Document {doc_id} (คะแนน TF-IDF: `{score:.4f}`)\n\n{documents[doc_id-1]}")
    else:
        st.warning("ไม่พบเอกสารที่เกี่ยวข้อง")
