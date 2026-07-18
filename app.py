import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# ================= PENGATURAN HALAMAN =================
st.set_page_config(page_title="Defect Analytics", page_icon="🏭", layout="wide")
st.title("🏭 Defect Analytics Dashboard")
st.markdown("Project Kecerdasan Buatan - Analisis Cacat Produk Manufaktur")
st.markdown("---")

# ================= GENERATE DATASET =================
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
    # Mengubah teks Severity menjadi angka (0, 1, 2) untuk grafik Y-Axis
    df['Severity_Code'] = df['SEVERITY'].map({'Minor': 0, 'Moderate': 1, 'Critical': 2})
    return df

df = load_data()

# ================= MENU TABS =================
tab1, tab2, tab3 = st.tabs(["📊 Hasil Clustering", "💡 Insights Bisnis", "🗃️ EDA & Dataset"])

# ================= KONTEN TAB 1 =================
with tab1:
    # Membuat Layout Slider dan Tombol Sejajar
    col_slider, col_btn = st.columns([4, 1])
    with col_slider:
        k_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=6, value=3, step=1)
    with col_btn:
        st.write("") # Spasi kosong agar tombol turun sejajar dengan slider
        st.write("")
        st.button("Proses Clustering", type="primary", use_container_width=True)

    # Menjalankan Algoritma K-Means
    X = df[['COST ($)', 'Severity_Code']]
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df['Cluster_Num'] = kmeans.fit_predict(X) + 1
    df['Cluster'] = "Cluster " + df['Cluster_Num'].astype(str)

    st.markdown("<br>", unsafe_allow_html=True)

    # Membuat Layout 2 Kolom untuk Grafik (Kiri 60%, Kanan 40%)
    col_plot1, col_plot2 = st.columns([3, 2])
    
    with col_plot1:
        st.subheader("Visualisasi Cluster")
        st.write("Biaya Perbaikan vs Tingkat Keparahan")
        
        fig1 = px.scatter(df, x="COST ($)", y="Severity_Code", color="Cluster",
                          hover_data=['TYPE', 'LOCATION', 'SEVERITY', 'INSPECTION'],
                          labels={"Severity_Code": "Severity"},
                          category_orders={"Cluster": [f"Cluster {i}" for i in range(1, 7)]})
        # Mengatur agar sumbu Y hanya menampilkan angka 0, 1, dan 2
        fig1.update_layout(yaxis=dict(tickmode='linear', tick0=0, dtick=1))
        fig1.update_traces(marker=dict(size=12, opacity=0.8))
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_plot2:
        st.subheader("Distribusi Data")
        st.write("Jumlah item per cluster")
        
        # Menghitung jumlah data per cluster untuk diagram batang
        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        cluster_counts = cluster_counts.sort_values('Cluster')
        
        fig2 = px.bar(cluster_counts, x='Cluster', y='Count', color='Cluster',
                      category_orders={"Cluster": [f"Cluster {i}" for i in range(1, 7)]})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ================= KONTEN TAB 2 =================
with tab2:
    st.subheader("Ringkasan Karakteristik Cluster")
    
    # Menghitung otomatis nilai rata-rata dan modus tiap cluster untuk Tabel
    summary_data = []
    for i in range(1, k_clusters + 1):
        c_data = df[df['Cluster_Num'] == i]
        summary_data.append({
            'CLUSTER': f"Cluster {i}",
            'CACAT DOMINAN': c_data['TYPE'].mode()[0],
            'SEVERITY': c_data['SEVERITY'].mode()[0],
            'LOKASI': c_data['LOCATION'].mode()[0],
            'INSPEKSI': c_data['INSPECTION'].mode()[0],
            'BIAYA RATA-RATA': f"${c_data['COST ($)'].mean():.2f}"
        })
    
    # Menampilkan Tabel Ringkasan
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Membuat 2 Kolom untuk Kotak Penjelasan
    col_text1, col_text2 = st.columns(2)
    with col_text1:
        st.info("🧠 **Penjelasan Model**\n\nAlgoritma **K-Means Clustering** digunakan untuk mengelompokkan data cacat berdasarkan kesamaan karakteristik (Tipe Cacat, Lokasi, Severity, Metode Inspeksi, dan Biaya Perbaikan). Proses ini mempermudah identifikasi kelompok kerusakan yang paling menguras anggaran perusahaan.")
        
    with col_text2:
        st.warning("💡 **Insights Manufaktur**\n\n*   **Alokasi Anggaran:** Fokus pengendalian kualitas dan pemeliharaan mesin pada cluster dengan biaya perbaikan rata-rata tertinggi.\n*   **Optimasi Inspeksi:** Cacat 'Critical' yang saat ini masih dideteksi secara 'Manual' harus segera beralih ke 'Automated' agar tidak lolos ke pelanggan.\n*   **Perbaikan Proses:** Evaluasi stasiun kerja yang secara konsisten menghasilkan cacat pada lokasi dominan di tiap cluster.")

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
    
    # Menampilkan dataset asli tanpa kolom tambahan hasil algoritma
    st.dataframe(df.drop(columns=['Severity_Code', 'Cluster_Num', 'Cluster'], errors='ignore'), use_container_width=True)
