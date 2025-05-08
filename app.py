import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("Customer Segmentation using Pre-trained K-Means Model")

# Load model
with open("K-Mean.pkl", "rb") as file:
    model = pickle.load(file)

# Load raw CSV data
df = pd.read_csv("clean_sales_data.csv")  # <-- Replace with your real file

# Preprocess the same way as in training
customer_df = df.groupby('Purchase Address').agg({
    'Order ID': 'nunique',
    'Sales': 'sum'
}).rename(columns={'Order ID': 'Frequency', 'Sales': 'Monetary'})

# Scale the features like during training
scaler = StandardScaler()
scaled = scaler.fit_transform(customer_df)

# Predict with model
labels = model.predict(scaled)
customer_df['Cluster'] = labels

# Reduce to 2D for plotting
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)

# Plotting
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=100, edgecolors='k')
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)
plt.title("K-Means Clusters (PCA Reduced)")
st.pyplot(fig)

# Show table
st.subheader("Clustered Customers")
st.write(customer_df)
