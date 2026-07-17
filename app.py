import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dashboard Clustering Manufaktur", layout="wide")
st.title("Analisis Clustering Cacat Produk Industri Manufaktur")
st.write("Aplikasi ini dibuat untuk memenuhi tugas UAS Deployment & Streamlit. Menggunakan data simulasi karena akses materi Pertemuan 12 terkendala.")

np.random.seed(42)
data = {
    'Suhu_Mesin_C': np.random.normal(loc=75, scale=10, size=200),
    'Kecepatan_Produksi': np.random.normal(loc=100, scale=15, size=200),
    'Tingkat_Kecacatan_Persen': np.random.normal(loc=5, scale=2, size=200)
}
df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Suhu_Mesin_C', 'Tingkat_Kecacatan_Persen']])

cluster_mapping = {0: 'Risiko Rendah', 1: 'Risiko Menengah', 2: 'Risiko Tinggi'}
df['Kategori_Risiko'] = df['Cluster'].map(cluster_mapping)

st.header("1. Dataset Cacat Produk (Simulasi)")
st.dataframe(df.head(10))

st.header("2. Visualisasi Hasil Clustering")
fig, ax = plt.subplots(figsize=(8, 5))
colors = {'Risiko Rendah': 'green', 'Risiko Menengah': 'orange', 'Risiko Tinggi': 'red'}

for kategori in colors.keys():
    subset = df[df['Kategori_Risiko'] == kategori]
    ax.scatter(subset['Suhu_Mesin_C'], subset['Tingkat_Kecacatan_Persen'], 
               c=colors[kategori], label=kategori, alpha=0.7, edgecolors='w')

ax.set_xlabel("Suhu Mesin (°C)")
ax.set_ylabel("Tingkat Kecacatan (%)")
ax.set_title("Clustering Cacat Produk berdasarkan Suhu Mesin")
ax.legend()
st.pyplot(fig)

st.header("3. Interpretasi Hasil & Business Insights")
st.markdown("""
*   **Cluster Hijau (Risiko Rendah):** Kondisi ideal, kecacatan minim.
*   **Cluster Oranye (Risiko Menengah):** Suhu mulai tidak stabil, kecacatan sedang.
*   **Cluster Merah (Risiko Tinggi):** Suhu ekstrem, kecacatan sangat tinggi.
**Rekomendasi:** Segera lakukan kalibrasi mesin jika suhu mulai mendekati zona Oranye/Merah.
""")
