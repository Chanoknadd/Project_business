import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Customer Segmentation with K-Means")

# Load your raw data (modify the path if needed)
df = pd.read_csv("clean_sales_data.csv")

# Preprocess like during model training
customer_df = df.groupby('Purchase Address').agg({
    'Order ID': 'nunique',
    'Sales': 'sum'
}).rename(columns={'Order ID': 'Frequency', 'Sales': 'Monetary'})

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(customer_df)

# ---- Elbow Plot ----
sse = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    sse.append(km.inertia_)

st.subheader("Elbow Method (to find optimal k)")
fig1, ax1 = plt.subplots()
ax1.plot(K_range, sse, marker='o')
ax1.set_title("Elbow Method")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("SSE")
ax1.grid(True)
st.pyplot(fig1)

# ---- Cluster Assignment with k=4 ----
k = st.slider("Select number of clusters (k)", 2, 10, 4)
kmeans = KMeans(n_clusters=k, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(scaled)

# ---- Pairplot ----
st.subheader("Customer Segments (Pairplot)")
fig2 = sns.pairplot(customer_df.reset_index(), hue='Cluster', palette='tab10')
st.pyplot(fig2)

# ---- Cluster Means ----
st.subheader("Average Values per Cluster")
st.write(customer_df.groupby('Cluster').mean())
