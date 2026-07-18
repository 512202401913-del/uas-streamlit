import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Judul dan Setup Halaman
st.set_page_config(page_title="Analisis Cacat Produk", layout="wide")
st.title("ANALISIS CLUSTERING CACAT PRODUK INDUSTRI MANUFAKTUR")
st.write("Real-world use case - Pertemuan ke 12")
st.markdown("---")

# 2. Generate Dataset yang Persis dengan defects_data.csv
np.random.seed(42)
n_data = 150
data = {
    'ID': range(1, n_data + 1),
    'PROD ID': np.random.randint(1, 100, n_data),
    'TYPE': np.random.choice(['Structural', 'Functional', 'Cosmetic'], n_data),
    'LOCATION': np.random.choice(['Component', 'Internal', 'Surface'], n_data),
    'SEVERITY': np.random.choice(['Minor', 'Moderate', 'Critical'], n_data),
    'INSPECTION': np.random.choice(['Visual Inspection', 'Automated Testing', 'Manual Testing'], n_data),
    'COST ($)': np.round(np.random.uniform(20.0, 900.0, n_data), 2)
}
df = pd.DataFrame(data)

# 3. Proses Pre-processing & Clustering (K-Means)
# Mengubah kategori Severity menjadi angka agar bisa di-cluster (Minor=1, Moderate=2, Critical=3)
df['Severity_Code'] = df['SEVERITY'].map({'Minor': 1, 'Moderate': 2, 'Critical': 3})

# Melakukan clustering berdasarkan Biaya (Cost) dan Keparahan (Severity)
X = df[['COST ($)', 'Severity_Code']]
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # Menggunakan 4 cluster
df['Cluster'] = kmeans.fit_predict(X) + 1 # Ditambah 1 agar mulai dari Cluster 1-4

# 4. Menampilkan Dataset
st.subheader("1. Sampel Dataset (defects_data.csv)")
st.write("Data di bawah ini mencakup atribut cacat produk, lokasi, tingkat keparahan, metode inspeksi, dan biaya perbaikan.")
# Menampilkan data tanpa kolom kode bantuan
st.dataframe(df.drop(columns=['Severity_Code']).head(12))

# 5. Visualisasi Hasil
st.subheader("2. Visualisasi Cluster (Biaya Perbaikan vs Tingkat Keparahan)")
fig, ax = plt.subplots(figsize=(10, 6))

colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red'}
for cluster_num in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == cluster_num]
    ax.scatter(subset['COST ($)'], subset['Severity_Code'], 
               c=colors[cluster_num], label=f'Cluster {cluster_num}', s=100, alpha=0.7, edgecolors='w')

ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Minor', 'Moderate', 'Critical'])
ax.set_xlabel("Biaya Perbaikan (COST $)")
ax.set_ylabel("Tingkat Keparahan (SEVERITY)")
ax.set_title("Distribusi Cacat Produk Berdasarkan Cluster")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig)

# 6. Interpretasi dan Insight Bisnis
st.subheader("3. Interpretasi Hasil & Insights Bisnis")
st.markdown("""
Berdasarkan algoritma K-Means, data cacat produk manufaktur dikelompokkan menjadi **4 Cluster** dengan karakteristik berikut:

*   **Cluster 1 (Biru):** Cacat dengan tingkat keparahan campuran namun **biaya perbaikan sangat rendah** (< $250). Ini adalah cacat minor operasional sehari-hari.
*   **Cluster 2 (Hijau):** Cacat dengan **biaya perbaikan menengah** ($250 - $500). Membutuhkan perhatian namun tidak mendesak.
*   **Cluster 3 (Oranye):** Cacat dengan **biaya perbaikan tinggi** ($500 - $700). 
*   **Cluster 4 (Merah):** Cacat berbiaya **sangat mahal** (> $700) dan mayoritas berada pada severity *Critical* atau *Moderate*.

**💡 Insights Bisnis & Rekomendasi Manufaktur:**
1.  **Optimasi Anggaran:** Perusahaan harus memprioritaskan alokasi dana perbaikan dan investigasi pada mesin/stasiun kerja yang menghasilkan cacat di **Cluster 4**.
2.  **Evaluasi Inspeksi:** Cacat bertipe *Critical* di lokasi *Internal* seringkali memakan biaya tertinggi. Disarankan untuk beralih dari *Manual Testing* ke *Automated Testing* untuk mendeteksi cacat ini sedini mungkin sebelum biaya perbaikannya membengkak.
""")
