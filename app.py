import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="NLP: TF vs Embeddings", layout="wide")

# --- Course style (Cultural Data Analysis, University of Amsterdam) ---
CRIMSON, CRIMSON_DARK, INK, BLUE = "#BC0031", "#7E0021", "#1F1D21", "#3F6391"
FONT = '"Source Sans 3", "Source Sans Pro", Arial, sans-serif'

# colour scales for tables (matplotlib) and charts (plotly)
CMAP_SEQ = LinearSegmentedColormap.from_list("cda_seq", ["#FFFFFF", "#F6D2DB", CRIMSON])
CMAP_LIGHT = LinearSegmentedColormap.from_list("cda_light", ["#FFFFFF", "#F0C4CE"])  # counts: a soft tint, text stays dark
CMAP_DIV = LinearSegmentedColormap.from_list("cda_div", [BLUE, "#FFFFFF", CRIMSON])
SCALE_SIM = [[0, "#FFFFFF"], [0.5, "#F0C4CE"], [1, CRIMSON]]

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=Faustina:ital,wght@0,400;1,400&display=swap">
<style>
  html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{ font-family: {FONT} !important; }}
  [data-testid="stDecoration"] {{ background: {CRIMSON}; background-image: none; height: 6px; }}
  h1, h2, h3 {{ color: {INK}; font-weight: 700 !important; }}
  .cda-sub {{ font-family: Faustina, Georgia, serif; font-style: italic; color: #555555; margin: -0.6rem 0 1rem; }}
  .cda-note {{ border-left: 4px solid {CRIMSON}; background: #FBEEF1; padding: 0.6rem 0.9rem; border-radius: 0 6px 6px 0; margin: 0.5rem 0 1rem; color: {INK}; }}
  .cda-note.blue {{ border-left-color: {BLUE}; background: #EEF2F7; }}
  .stTabs [aria-selected="true"] {{ color: {CRIMSON}; }}
  [data-testid="stSidebar"] {{ border-right: 1px solid #BEC6D1; }}
  a, .stMarkdown a, [data-testid="stSidebar"] a {{ color: {CRIMSON} !important; }}
</style>
""", unsafe_allow_html=True)


def note(text, blue=False):
    st.markdown(f'<div class="cda-note{" blue" if blue else ""}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=None):
    fig.update_layout(font=dict(family=FONT, color=INK, size=14), title_font=dict(size=18),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", margin=dict(l=40, r=20, t=60, b=40))
    if height:
        fig.update_layout(height=height)
    return fig


# --- Helper Functions ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def plot_heatmap(df, title):
    fig = px.imshow(df, text_auto=".2f", aspect="auto", color_continuous_scale=SCALE_SIM, zmin=0, zmax=1, title=title)
    return style_fig(fig)


# --- Sidebar ---
st.sidebar.header("Configuration")

default_sentences = """The quick brown fox jumps over the lazy dog.
A fast brown fox leaps over a sleepy canine.
I love machine learning and natural language processing.
Artificial intelligence is fascinating.
The weather is nice today."""

user_input = st.sidebar.text_area("Enter sentences (one per line):", value=default_sentences, height=150)
sentences = [s.strip() for s in user_input.split('\n') if s.strip()]
st.sidebar.markdown("Part of the [Cultural Data Analysis teaching tools](https://goto4711.github.io/cda-teaching-tools/tools/embeddings/).")

# --- Main Content ---
st.title("NLP Visualization: Term Frequency vs. Modern Embeddings")
st.markdown('<p class="cda-sub">Cultural Data Analysis &middot; Week 4: Cultural Data Forms: Text</p>', unsafe_allow_html=True)
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
    st.dataframe(df_tf.style.background_gradient(cmap=CMAP_LIGHT, vmin=0))

    st.subheader("Similarity Matrix (Based on Word Overlap)")
    sim_tf = cosine_similarity(X_tf)
    df_sim_tf = pd.DataFrame(sim_tf, index=[f"S{i+1}" for i in range(len(sentences))], columns=[f"S{i+1}" for i in range(len(sentences))])
    st.plotly_chart(plot_heatmap(df_sim_tf, "Cosine Similarity (TF)"), use_container_width=True)
    note("Notice: If two sentences have no common words, their similarity is 0, even if they mean the same thing!")

# --- Tab 2: Embeddings ---
with tab2:
    st.header("Modern Dense Embeddings")
    st.write(f"Each sentence is converted into a vector of size {embeddings.shape[1]}. Here are the first 20 dimensions "
             "(blue-grey below zero, crimson above):")
    lim = float(np.abs(df_emb.iloc[:, :20].values).max())
    st.dataframe(df_emb.iloc[:, :20].style.background_gradient(cmap=CMAP_DIV, vmin=-lim, vmax=lim).format("{:.3f}"))

    st.subheader("Similarity Matrix (Semantic)")
    sim_emb = cosine_similarity(embeddings)
    df_sim_emb = pd.DataFrame(sim_emb, index=[f"S{i+1}" for i in range(len(sentences))], columns=[f"S{i+1}" for i in range(len(sentences))])
    st.plotly_chart(plot_heatmap(df_sim_emb, "Cosine Similarity (Embeddings)"), use_container_width=True)
    note("Notice: Sentences with similar meanings (e.g., 'dog' and 'canine') have high similarity scores, even without shared words.", blue=True)

# --- Tab 3: PCA Visualization ---
with tab3:
    st.header("2D Projection (PCA)")
    st.write(f"We use PCA to reduce the {embeddings.shape[1]}-dimensional vectors down to 2 dimensions so we can plot them.")

    pca = PCA(n_components=2)
    components = pca.fit_transform(embeddings)

    df_pca = pd.DataFrame(components, columns=['x', 'y'])
    df_pca['sentence'] = sentences

    fig_pca = px.scatter(df_pca, x='x', y='y', text='sentence', title="Sentence Embeddings in 2D Space")
    fig_pca.update_traces(textposition='top center', marker=dict(size=14, color=CRIMSON, line=dict(width=1.5, color="#FFFFFF")))
    fig_pca.update_xaxes(showgrid=True, gridcolor="#EBEBEC", zerolinecolor="#BEC6D1")
    fig_pca.update_yaxes(showgrid=True, gridcolor="#EBEBEC", zerolinecolor="#BEC6D1")
    st.plotly_chart(style_fig(fig_pca, height=600), use_container_width=True)

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
        st.dataframe(results_df.sort_values(by='Semantic Score (Emb)', ascending=False)
                     .style.background_gradient(subset=['Keyword Score (TF)', 'Semantic Score (Emb)'], cmap=CMAP_SEQ, vmin=0, vmax=1)
                     .format({'Keyword Score (TF)': "{:.2f}", 'Semantic Score (Emb)': "{:.2f}"}))

        best_tf = results_df.loc[results_df['Keyword Score (TF)'].idxmax()]
        best_emb = results_df.loc[results_df['Semantic Score (Emb)'].idxmax()]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Keyword Match", f"{best_tf['Keyword Score (TF)']:.2f}")
            st.caption(best_tf['Sentence'] if best_tf['Keyword Score (TF)'] > 0 else "No sentence shares a word with the query.")
        with col2:
            st.metric("Best Semantic Match", f"{best_emb['Semantic Score (Emb)']:.2f}")
            st.caption(best_emb['Sentence'])
