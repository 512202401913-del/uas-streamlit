import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# 1. Judul dan Setup Halaman (Mode Lebar)
st.set_page_config(page_title="Defect Analytics", page_icon="🏭", layout="wide")
st.title("🏭 Defect Analytics Dashboard")
st.markdown("Kelompokkan data cacat produk untuk menemukan pola kerusakan dan optimasi biaya. *(Real-world use case - Pertemuan 12)*")
st.markdown("---")

# 2. Generate Dataset Asli
@st.cache_data
def load_data():
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
    df['Severity_Code'] = df['SEVERITY'].map({'Minor': 1, 'Moderate': 2, 'Critical': 3})
    return df

df = load_data()

# 3. MEMBUAT MENU TABS (Bisa Dipencet-pencet)
tab1, tab2, tab3 = st.tabs(["📊 Hasil Clustering", "💡 Insights Bisnis", "🗃️ EDA & Dataset"])

# ================= KONTEN TAB 1 =================
with tab1:
    st.subheader("Konfigurasi Model")
    # Slider interaktif untuk memilih jumlah cluster
    k_clusters = st.slider("Tentukan Jumlah Cluster (K)", min_value=2, max_value=6, value=4, step=1)
    
    # Proses K-Means
    X = df[['COST ($)', 'Severity_Code']]
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X) + 1
    df['Cluster'] = "Cluster " + df['Cluster'].astype(str) # Ubah format nama cluster
    
    st.markdown("---")
    st.subheader("Visualisasi Cluster")
    
    # Membuat Kartu Metrik (Angka Besar)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sampel Cacat", f"{len(df)} Kasus")
    col2.metric("Jumlah Cluster Aktif", k_clusters)
    col3.metric("Rata-rata Biaya", f"${df['COST ($)'].mean():.2f}")
    
    # Grafik Interaktif dengan Plotly
    fig = px.scatter(df, x="COST ($)", y="SEVERITY", color="Cluster",
                     title="Biaya Perbaikan vs Tingkat Keparahan",
                     hover_data=['TYPE', 'LOCATION', 'INSPECTION'],
                     category_orders={"SEVERITY": ["Minor", "Moderate", "Critical"]})
    fig.update_traces(marker=dict(size=14, opacity=0.8, line=dict(width=1, color='white')))
    st.plotly_chart(fig, use_container_width=True)

# ================= KONTEN TAB 2 =================
with tab2:
    st.subheader("Interpretasi & Insights Bisnis")
    st.info("**Penjelasan Model:** Algoritma K-Means Clustering digunakan untuk mengelompokkan data cacat berdasarkan kesamaan karakteristik biaya perbaikan dan tingkat keparahan (Severity).")
    
    st.markdown("""
    **💡 Insights Manufaktur & Rekomendasi:**
    *   **Alokasi Anggaran:** Fokus pengendalian kualitas dan dana perbaikan mesin harus diprioritaskan pada cluster dengan *Cost* tertinggi dan *Severity Critical* (Cluster Merah/Berbiaya Tinggi).
    *   **Optimasi Inspeksi:** Cacat *Critical* yang saat ini masih dideteksi melalui *Manual Testing* sangat berisiko lolos ke tangan konsumen karena *human error*. Sangat disarankan beralih ke *Automated Testing*.
    *   **Perbaikan Proses:** Mesin atau stasiun kerja yang menghasilkan cacat dominan tipe *Structural* pada lokasi *Internal* perlu dievaluasi ulang penjadwalan perawatannya (*Preventive Maintenance*).
    """)

# ================= KONTEN TAB 3 =================
with tab3:
    st.subheader("Eksplorasi Data Awal (EDA)")
    
    # Membagi layar jadi 2 grafik bersebelahan
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.write("**Distribusi Tipe Cacat**")
        fig_bar = px.histogram(df, x="TYPE", color="SEVERITY", barmode="group")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_eda2:
        st.write("**Metode Inspeksi yang Digunakan**")
        fig_pie = px.pie(df, names="INSPECTION", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Sampel Dataset Asli (defects_data.csv)")
    st.dataframe(df.drop(columns=['Severity_Code']), use_container_width=True)
