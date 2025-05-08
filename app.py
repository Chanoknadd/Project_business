import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.title("Customer Segmentation using Pre-trained K-Means Model")

# Load model
with open("K-Mean.pkl", "wb") as file:
    model = pickle.load(file)

# Load input data from a fixed CSV file
data_path = "clean_sales_data.csv"  # <- CHANGE this to your actual path
df = pd.read_csv(data_path)

# Display raw data
st.subheader("Raw Input Data")
st.write(df.head())

# Predict clusters
labels = model.predict(df)
df['Cluster'] = labels

# Dimensionality reduction for plotting (if data has many columns)
pca = PCA(n_components=2)
reduced = pca.fit_transform(df.drop(columns=['Cluster']))

# Plotting
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=100, edgecolors='k')
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)
plt.title("K-Means Clusters (PCA Reduced)")
st.pyplot(fig)

# Show clustered data
st.subheader("Clustered Data")
st.write(df)
