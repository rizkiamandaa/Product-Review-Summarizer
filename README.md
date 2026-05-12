# Product-Review-Summarizer
An AI-powered system that automatically summarizes Indonesian e-commerce product reviews into a single, coherent paragraph — helping sellers and product teams understand customer sentiment without reading hundreds of reviews manually.
'Built as an upgrade to my undergraduate thesis (Peringkasan Ulasan Produk Rumah Tangga Menggunakan BERTopic dan Maximal Marginal Relevance, Universitas Sumatera Utara, 2024), replacing extractive MMR summarization with abstractive LLM generation.'

# How It Works
CSV Upload (Indonesian reviews)
       ↓
Minimal Preprocessing
(noise removal only — no stemming/stopwords)
       ↓
IndoBERT Sentiment Classification
(positive / negative / neutral per sentence)
       ↓
BERTopic Topic Modeling (per sentiment)
(SBERT embeddings → UMAP → HDBSCAN → c-TF-IDF)
       ↓
Gemini LLM Summarization
(all aspects + sentiment distribution → 1 unified paragraph)
       ↓
Auto ROUGE Evaluation
(BERTopic representative sentences as pseudo-reference)

Why no stemming or stopword removal?
IndoBERT and BERTopic both use transformer-based contextual embeddings trained on natural text. Removing stopwords before feeding them actually hurts performance — for example, "tidak bagus" (not good) becomes "bagus" (good) after stopword removal, causing IndoBERT to misclassify it as positive.

# Tech Stack
<table>
  <ch>Component</ch>
    <cl>Sentiment Analysis</cl>
    <cl>Topic Modeling</cl>
    <cl>Embeddings</cl>
    <cl>Summarization</cl>
    <cl>Evaluation</cl>
    <cl>UI</cl>
  <ch>Model / Library</ch>
    <cl>IndoBERT</cl>
    <cl>BERTopic</cl>
    <cl>paraphrase-multilingual-MiniLM-L12-v2</cl>
    <cl>Gemini 1.5 Flash (REST API)</cl>
    <cl>ROUGE-1, ROUGE-2, ROUGE-L>
    <cl>Streamlit</cl>
  <ch>Purpose</ch>
    <cl>Classify each review sentence as positive / negative / neutral</cl>
    <cl>Discover key aspects from review clusters</cl>
    <cl>Multilingual sentence embeddings (supports Indonesian)</cl>
    <cl>Generate abstractive unified paragraph>
    <cl>Measure summary quality (auto, no manual reference)</cl>
    <cl>Interactive web interface</cl>
</table>

# Getting Started
1. Clone the repo
"git clone https://github.com/yourusername/product-review-summarizer.git
cd product-review-summarizer"

2. Install Dependences
"pip install -r requirements.txt"

3. Setup Your API Key
Create a .env file in the root directory:
"GEMINI_API_KEY=your_api_key_here"

4. Run The App
"streamlit run shopee_review_sum.py"

# Input Format
Upload a .csv file with at least one column containing review text in Indonesian. Example:
<bold>comments</bold>
Produk bagus banget, pengiriman cepat, seller ramah
Kualitas oke tapi agak bising kalau dipakai lama
Sudah pakai 2 minggu, berfungsi dengan baik semoga awet.

