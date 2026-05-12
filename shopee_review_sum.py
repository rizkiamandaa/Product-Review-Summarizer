import re
import os
import requests
import pandas as pd
import streamlit as st
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from rouge_score import rouge_scorer


# CONFIGURATION

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def list_available_models() -> list:
    """Check available models in this API key"""
    url  = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return [f"ERROR {resp.status_code}: {resp.text[:200]}"]
    models = resp.json().get("models", [])
    return [
        m["name"] for m in models
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def get_first_flash_model() -> str:
    """
    Auto-detect available model flash for this API key.
    """
    models = list_available_models()
    for m in models:
        if "flash" in m.lower():
            return m.replace("models/", "")
    # Fallback to first model if there is no flash
    if models and not models[0].startswith("ERROR"):
        return models[0].replace("models/", "")
    raise RuntimeError(f"There is no available Gemini model. Detail: {models}")


# 1. PREPROCESSING (remove some noises)

def clean_text(text: str) -> str:
    # Remove URL
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove HTML tag
    text = re.sub(r"<[^>]+>", "", text)
    # Normalization
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Remove non-alfabet except spaces and basic punctuation
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    # Normalization of excessive space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Add clean_review column to dataframe.
    """
    df = df.dropna(subset=[col]).copy()
    df["clean_review"] = df[col].astype(str).apply(clean_text)
    # Remove reviews <5 words after cleaning
    df = df[df["clean_review"].apply(lambda x: len(x.split()) >= 5)]
    return df.reset_index(drop=True)


# 2. SENTIMENT CLASSIFICATION — IndoBERT

@st.cache_resource(show_spinner="Load IndoBERT...")
def load_sa_model():
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tok,
        device=device,
    )

# LABEL_0 = positif, LABEL_1 = netral, LABEL_2 = negatif
LABEL_MAP = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}


def classify_sentiments(texts: list, batch_size: int = 32) -> list:
    sa = load_sa_model()
    results = []
    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        outputs = sa(batch, truncation=True, max_length=128)
        results.extend([LABEL_MAP[o["label"]] for o in outputs])
    return results

# 3. TOPIC MODELING — BERTopic

@st.cache_resource(show_spinner="Create embedding model...")
def load_embedding_model():
    # Model multilingual (Bahasa Indonesia)
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def run_bertopic(docs: list, min_topic_size: int = 3) -> dict:
    """
    SBERT in BERTopic has handled semantic by itself.
    """
    if len(docs) < 5:
        return {"Umum": docs[:3]}

    embedding_model = load_embedding_model()
    topic_model     = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=False,
        calculate_probabilities=False,
    )

    topics, _ = topic_model.fit_transform(docs)

    result     = {}
    topic_info = topic_model.get_topic_info()

    # Sort based on document number (Count), top 5 aspect
    topic_info = topic_info[topic_info["Topic"] != -1].nlargest(5, "Count")

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        keywords = topic_model.get_topic(topic_id)
        label    = " | ".join([kw for kw, _ in keywords[:2]])  # 2. cleaner keyword

        rep_docs      = topic_model.get_representative_docs(topic_id)
        result[label] = [d for d in rep_docs if len(d.split()) >= 3][:3]

    return result if result else {"Umum": docs[:3]}


# 4. LLM SUMMARIZATION — Gemini 1.5 Flash

def build_prompt(
    product_name: str,
    pct_pos: int,
    pct_neg: int,
    aspek_pos: dict,
    aspek_neg: dict,
) -> str:
    """
    Create a complete prompt from Gemini.
    Include : sentiment distribution + aspects + original review sentence.
    """
    pos_section = ""
    for label, kalimat_list in aspek_pos.items():
        if kalimat_list:
            contoh = "; ".join(f'"{k[:100]}"' for k in kalimat_list[:2])
            pos_section += f"  • {label}: {contoh}\n"

    neg_section = ""
    for label, kalimat_list in aspek_neg.items():
        if kalimat_list:
            contoh = "; ".join(f'"{k[:100]}"' for k in kalimat_list[:2])
            neg_section += f"  • {label}: {contoh}\n"

    if not neg_section.strip():
        neg_section = "  • (tidak ada keluhan signifikan)\n"

    return f"""Kamu adalah sistem peringkasan ulasan produk e-commerce Indonesia.

DATA REVIEW:
Produk   : {product_name}
Sentimen : {pct_pos}% positif, {pct_neg}% negatif

ASPEK YANG DISUKAI PEMBELI:
{pos_section}
ASPEK YANG DIKELUHKAN PEMBELI:
{neg_section}

INSTRUKSI:
Tulis ringkasan produk dalam TEPAT 3-4 kalimat Bahasa Indonesia yang natural dan mengalir.

Aturan:
1. Kalimat 1 — kesan umum, nada cerminkan distribusi ({pct_pos}% positif)
2. Kalimat 2 — keunggulan utama yang paling sering dipuji
3. Kalimat 3 — kelemahan atau catatan penting untuk calon pembeli
4. Kalimat 4 (opsional) — rekomendasi atau konteks tambahan
5. JANGAN copy-paste kalimat review
6. JANGAN tulis pembuka seperti "Berikut ringkasan..." — langsung paragrafnya"""


def generate_summary(
    product_name: str,
    pct_pos: int,
    pct_neg: int,
    aspek_pos: dict,
    aspek_neg: dict,
) -> str:
    prompt   = build_prompt(product_name, pct_pos, pct_neg, aspek_pos, aspek_neg)
    model_name = get_first_flash_model()
    url        = (
        f"https://generativelanguage.googleapis.com"
        f"/v1beta/models/{model_name}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp    = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# 5. EVALUATION — ROUGE

def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    s = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": {"Precision": round(s["rouge1"].precision, 4),
                    "Recall":    round(s["rouge1"].recall, 4),
                    "F1":        round(s["rouge1"].fmeasure, 4)},
        "ROUGE-2": {"Precision": round(s["rouge2"].precision, 4),
                    "Recall":    round(s["rouge2"].recall, 4),
                    "F1":        round(s["rouge2"].fmeasure, 4)},
        "ROUGE-L": {"Precision": round(s["rougeL"].precision, 4),
                    "Recall":    round(s["rougeL"].recall, 4),
                    "F1":        round(s["rougeL"].fmeasure, 4)},
    }

# 6. PIPELINE

def run_pipeline(reviews_raw: list, product_name: str) -> dict:
    """
    Full pipeline from raw list review to final summarization.
    """
    # ── Preprocessing minimal ───────────────────────────
    df = pd.DataFrame({"review_asli": reviews_raw})
    df = preprocess_dataframe(df, "review_asli")

    # ── Sentiment classification ─────────────────────────
    df["sentimen"] = classify_sentiments(df["clean_review"].tolist())
    dist    = df["sentimen"].value_counts(normalize=True).to_dict()
    pct_pos = round(dist.get("positif", 0) * 100)
    pct_neg = round(dist.get("negatif", 0) * 100)
    pct_neu = round(dist.get("netral",  0) * 100)

    # ── BERTopic ─────────────────────────────────────────
    docs_pos = df[df["sentimen"] == "positif"]["review_asli"].tolist()
    docs_neg = df[df["sentimen"] == "negatif"]["review_asli"].tolist()

    aspek_pos = run_bertopic(docs_pos) if len(docs_pos) >= 5 else {}
    aspek_neg = run_bertopic(docs_neg) if len(docs_neg) >= 5 else {}

    # ── LLM summarization ───────────────────────────────
    ringkasan = generate_summary(
        product_name, pct_pos, pct_neg, aspek_pos, aspek_neg
    )

    # ── Auto ROUGE — use representatif sentence as pseudo-reference ──
    # Join all the representative sentence from BERTopic
    rep_sentences = []
    for kalimat_list in aspek_pos.values():
        rep_sentences.extend(kalimat_list)
    for kalimat_list in aspek_neg.values():
        rep_sentences.extend(kalimat_list)
    pseudo_ref = " ".join(rep_sentences) if rep_sentences else " ".join(df["clean_review"].tolist()[:30])
    auto_rouge = compute_rouge(ringkasan, pseudo_ref)

    return {
        "df":         df,
        "total":      len(df),
        "pct_pos":    pct_pos,
        "pct_neg":    pct_neg,
        "pct_neu":    pct_neu,
        "aspek_pos":  aspek_pos,
        "aspek_neg":  aspek_neg,
        "ringkasan":  ringkasan,
        "auto_rouge": auto_rouge,
    }


# 7. STREAMLIT UI

def main():
    st.set_page_config(
        page_title="Shopee Review Summarizer",
        page_icon="🛒",
        layout="wide",
    )

    st.title("🛒 Shopee Review Summarizer")
    st.caption("IndoBERT · BERTopic · Gemini 1.5 Flash")
    st.divider()

    # ── Input ────────────────────────────────────────────
    st.subheader("1. Upload Review Data")

    reviews_raw  = []
    product_name = ""

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df_raw     = pd.read_csv(uploaded)
        review_col = st.selectbox("Review column:", df_raw.columns.tolist())
        product_name = st.text_input("Product name", placeholder="e.g. Humidifier")
        st.dataframe(df_raw[[review_col]].head(3), use_container_width=True)

        if st.button("▶️ Analyze", type="primary"):
            if not product_name:
                st.error("Please enter the product name.")
            else:
                reviews_raw = df_raw[review_col].dropna().astype(str).tolist()
                st.success(f"✅ {len(reviews_raw)} reviews loaded.")

    st.divider()

    # ── Pipeline ─────────────────────────────────────────
    if reviews_raw and product_name:
        with st.spinner("Running analysis pipeline... (1-3 minutes)"):
            hasil = run_pipeline(reviews_raw, product_name)

        # ── Metrics ──────────────────────────────────────
        st.subheader("2. Review Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews", hasil["total"])
        c2.metric("Positive",      f"{hasil['pct_pos']}%")
        c3.metric("Negative",      f"{hasil['pct_neg']}%")
        c4.metric("Neutral",       f"{hasil['pct_neu']}%")

        # ── Aspect ────────────────────────────────────────
        st.subheader("3. Product Aspects (BERTopic)")
        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("**✅ Positive Aspects**")
            for label, kalimat_list in hasil["aspek_pos"].items():
                with st.expander(f"📌 {label}"):
                    for k in kalimat_list:
                        st.caption(f"› {k}")
            if not hasil["aspek_pos"]:
                st.caption("Not enough data for clustering.")

        with col_neg:
            st.markdown("**⚠️ Negative Aspects**")
            for label, kalimat_list in hasil["aspek_neg"].items():
                with st.expander(f"📌 {label}"):
                    for k in kalimat_list:
                        st.caption(f"› {k}")
            if not hasil["aspek_neg"]:
                st.caption("Not enough data for clustering.")

        # ── FINAL OUTPUT ─────────────────────────────────
        st.subheader("4. ✨ Product Summary")
        st.info(
            f"This summary was generated by AI from {hasil['total']} Shopee buyer reviews."
        )
        st.markdown(
            f"""<div style="
                border-left: 4px solid #00d4aa;
                border-radius: 0 8px 8px 0;
                padding: 18px 22px;
                background: rgba(0, 212, 170, 0.05);
                font-size: 15px;
                line-height: 1.85;
            ">{hasil['ringkasan']}</div>""",
            unsafe_allow_html=True,
        )

        # ── ROUGE ────────────────────────────────────────
        st.subheader("5. ROUGE Evaluation")

        # Auto ROUGE ──────────────────────────────────────
        st.markdown("**📊 Auto ROUGE** — summary vs. original reviews")
        st.caption(
            "Measures how much review content is captured in the summary. "
            "No manual reference needed — original reviews serve as ground truth."
        )
        auto_df = (
            pd.DataFrame(hasil["auto_rouge"])
            .T.reset_index()
            .rename(columns={"index": "Metric"})
        )
        st.dataframe(auto_df, use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ Note: ROUGE is designed for extractive summarization. "
            "For abstractive (LLM), low scores are normal — "
            "the LLM generates new words rather than copying from reviews. "
            "What matters: Recall > 0 means review content is captured in the summary."
        )

        # ── Detail review ─────────────────────────────────
        with st.expander("📋 All reviews + sentiment"):
            st.dataframe(
                hasil["df"][["review_asli", "clean_review", "sentimen"]],
                use_container_width=True,
                height=300,
            )


if __name__ == "__main__":
    main()