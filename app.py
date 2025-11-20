import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="NLP: TF vs Embeddings", layout="wide")

# --- Helper Functions ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def plot_heatmap(df, title):
    fig = px.imshow(df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title=title)
    return fig

# --- Sidebar ---
st.sidebar.header("Configuration")

default_sentences = """The quick brown fox jumps over the lazy dog.
A fast brown fox leaps over a sleepy canine.
I love machine learning and natural language processing.
Artificial intelligence is fascinating.
The weather is nice today."""

user_input = st.sidebar.text_area("Enter sentences (one per line):", value=default_sentences, height=150)
sentences = [s.strip() for s in user_input.split('\n') if s.strip()]

# --- Main Content ---
st.title("NLP Visualization: Term Frequency vs. Modern Embeddings")
st.markdown("""
This tool visualizes the difference between **Term Frequency (Bag of Words)** representations and **Modern Dense Embeddings**.
*   **TF (Term Frequency)**: Counts how often words appear. Good for keyword matching, but misses meaning (e.g., "dog" vs "canine").
*   **Embeddings**: Captures semantic meaning in a vector space. "Dog" and "canine" will be close together.
""")

tab1, tab2, tab3, tab4 = st.tabs(["1. Term Frequency (TF)", "2. Modern Embeddings", "3. Visualization (PCA)", "4. Semantic Search Demo"])

# --- Logic ---

# 1. Term Frequency
vectorizer = CountVectorizer()
X_tf = vectorizer.fit_transform(sentences)
df_tf = pd.DataFrame(X_tf.toarray(), columns=vectorizer.get_feature_names_out(), index=[f"Sent {i+1}" for i in range(len(sentences))])

# 2. Embeddings
model = load_model()
embeddings = model.encode(sentences)
df_emb = pd.DataFrame(embeddings, index=[f"Sent {i+1}" for i in range(len(sentences))])


# --- Tab 1: Term Frequency ---
with tab1:
    st.header("Term Frequency (Bag of Words)")
    st.write("This matrix shows the count of each word in each sentence. Notice how sparse (lots of zeros) it can be.")
    st.dataframe(df_tf.style.background_gradient(cmap='Blues'))
    
    st.subheader("Similarity Matrix (Based on Word Overlap)")
    sim_tf = cosine_similarity(X_tf)
    df_sim_tf = pd.DataFrame(sim_tf, index=[f"S{i+1}" for i in range(len(sentences))], columns=[f"S{i+1}" for i in range(len(sentences))])
    st.plotly_chart(plot_heatmap(df_sim_tf, "Cosine Similarity (TF)"), use_container_width=True)
    st.info("Notice: If two sentences have no common words, their similarity is 0, even if they mean the same thing!")

# --- Tab 2: Embeddings ---
with tab2:
    st.header("Modern Dense Embeddings")
    st.write(f"Each sentence is converted into a vector of size {embeddings.shape[1]}. Here are the first 20 dimensions:")
    st.dataframe(df_emb.iloc[:, :20].style.background_gradient(cmap='RdBu_r'))
    
    st.subheader("Similarity Matrix (Semantic)")
    sim_emb = cosine_similarity(embeddings)
    df_sim_emb = pd.DataFrame(sim_emb, index=[f"S{i+1}" for i in range(len(sentences))], columns=[f"S{i+1}" for i in range(len(sentences))])
    st.plotly_chart(plot_heatmap(df_sim_emb, "Cosine Similarity (Embeddings)"), use_container_width=True)
    st.success("Notice: Sentences with similar meanings (e.g., 'dog' and 'canine') have high similarity scores, even without shared words.")

# --- Tab 3: PCA Visualization ---
with tab3:
    st.header("2D Projection (PCA)")
    st.write("We use PCA to reduce the 384-dimensional vectors down to 2 dimensions so we can plot them.")
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(embeddings)
    
    df_pca = pd.DataFrame(components, columns=['x', 'y'])
    df_pca['sentence'] = sentences
    
    fig_pca = px.scatter(df_pca, x='x', y='y', text='sentence', size_max=60, title="Sentence Embeddings in 2D Space")
    fig_pca.update_traces(textposition='top center')
    fig_pca.update_layout(height=600)
    st.plotly_chart(fig_pca, use_container_width=True)

# --- Tab 4: Search Demo ---
with tab4:
    st.header("Semantic Search vs Keyword Search")
    query = st.text_input("Enter a search query:", "puppy")
    
    if query:
        # TF Search
        query_vec_tf = vectorizer.transform([query])
        scores_tf = cosine_similarity(query_vec_tf, X_tf).flatten()
        
        # Embedding Search
        query_emb = model.encode([query])
        scores_emb = cosine_similarity(query_emb, embeddings).flatten()
        
        results_df = pd.DataFrame({
            'Sentence': sentences,
            'Keyword Score (TF)': scores_tf,
            'Semantic Score (Emb)': scores_emb
        })
        
        st.subheader("Results")
        st.dataframe(results_df.sort_values(by='Semantic Score (Emb)', ascending=False).style.background_gradient(subset=['Keyword Score (TF)', 'Semantic Score (Emb)'], cmap='Greens'))
        
        best_tf = results_df.loc[results_df['Keyword Score (TF)'].idxmax()]
        best_emb = results_df.loc[results_df['Semantic Score (Emb)'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Keyword Match", f"{best_tf['Keyword Score (TF)']:.2f}", best_tf['Sentence'])
        with col2:
            st.metric("Best Semantic Match", f"{best_emb['Semantic Score (Emb)']:.2f}", best_emb['Sentence'])
