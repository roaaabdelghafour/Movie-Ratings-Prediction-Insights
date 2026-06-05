import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================= Page Config =================
st.set_page_config(
    page_title="Movie Ratings & Revenue Dashboard",
    layout="wide"
)

st.title("🎬 Movie Ratings & Revenue Analytics Dashboard")
st.markdown("Interactive dashboard for EDA, statistical insights, and PCA visualization.")

# ================= Load Data =================

import pandas as pd

df = pd.read_csv(r"tmdb_cleaned_movies.csv")


# ================= Feature Engineering =================


# Safety checks
for col in ['budget', 'revenue']:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# Create features
df['profit'] = df['revenue'] - df['budget']
df['ROI'] = df['profit'] / df['budget']
df['ROI'].replace([np.inf, -np.inf], 0, inplace=True)
df['ROI'].fillna(0, inplace=True)

df['budget_log'] = np.log1p(df['budget'])
df['revenue_log'] = np.log1p(df['revenue'])

# ================= Sidebar =================
st.sidebar.header("Controls")

numeric_filter = st.sidebar.slider(
    "Minimum Vote Count",
    min_value=int(df['vote_count'].min()),
    max_value=int(df['vote_count'].max()),
    value=int(df['vote_count'].quantile(0.25))
)

df_filtered = df[df['vote_count'] >= numeric_filter]

# ================= Dataset Overview =================
st.header("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Movies", df_filtered.shape[0])
c2.metric("Average Rating", round(df_filtered['vote_average'].mean(), 2))
c3.metric("Average Revenue", f"${int(df_filtered['revenue'].mean()):,}")

st.dataframe(df_filtered.head())

# ================= EDA =================
st.header("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['vote_average'], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Movie Ratings")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='budget_log',
        y='revenue_log',
        data=df_filtered,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Budget vs Revenue (Log Scale)")
    st.pyplot(fig)

# ================= Correlation =================
st.header("🔗 Correlation Analysis")

corr_features = [
    'budget_log', 'revenue_log', 'profit',
    'ROI', 'vote_average', 'vote_count',
    'popularity', 'runtime'
]

corr = df_filtered[corr_features].corr()

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig)

# ================= PCA =================
st.header("🧠 PCA Visualization")

pca_features = [
    'budget_log', 'revenue_log', 'profit',
    'ROI', 'vote_average', 'vote_count',
    'popularity', 'runtime'
]

X = df_filtered[pca_features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

fig, ax = plt.subplots()
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.6, ax=ax)
ax.set_title("PCA Projection of Movies")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)

st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ================= Insights =================
st.header("💡 Key Insights")

st.markdown("""
- Movie ratings are more influenced by popularity and vote engagement than budget alone.
- High budgets do not guarantee higher ratings or ROI.
- PCA reveals latent structure separating large-scale blockbusters from smaller productions.
""")

st.success("Dashboard loaded successfully ✔")
