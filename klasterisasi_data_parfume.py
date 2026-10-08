# ============================================================
# PRAKTIK MANDIRI - UNSUPERVISED LEARNING
# DATASET PERFUME
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. MUAT DAN EKSPLORASI DATA
# ============================================================

print("=" * 60)
print("1. MUAT DAN EKSPLORASI DATA")
print("=" * 60)

# Membaca dataset
df = pd.read_excel("perfume_data.xlsx", header=None)

# Memberikan nama kolom
df.columns = ["perfume"] + [
    f"F{i}" for i in range(1, df.shape[1])
]

# Menampilkan 5 data pertama
print("\n--- HEAD ---")
print(df.head())

# Informasi dataset
print("\n--- INFO ---")
print(df.info())

# Statistik deskriptif
print("\n--- DESCRIBE ---")
print(df.describe())

# Ukuran dataset
print("\nJumlah baris :", df.shape[0])
print("Jumlah kolom:", df.shape[1])
print("Jumlah fitur :", df.shape[1] - 1)

# Mengecek missing value
print("\n--- MISSING VALUE ---")
print(df.isnull().sum())

print(
    "\nTotal missing value:",
    df.isnull().sum().sum()
)

# Mengecek data duplikat
print("\n--- DUPLIKAT ---")
print(
    "Jumlah data duplikat:",
    df.duplicated().sum()
)

# Menampilkan data duplikat
if df.duplicated().sum() > 0:
    print("\nData yang duplikat:")
    print(
        df[df.duplicated(keep=False)]
    )


# ============================================================
# 2. SCALING DAN MEMILIH k
# ============================================================

print("\n" + "=" * 60)
print("2. SCALING DAN MEMILIH k")
print("=" * 60)

# Kolom pertama adalah nama perfume
nama_perfume = df["perfume"]

# Mengambil fitur numerik
X = df.drop(columns=["perfume"])

# Memastikan semua fitur bertipe numerik
X = X.astype(float)

print("\nUkuran data sebelum scaling:")
print(X.shape)

# Standardisasi data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nUkuran data setelah scaling:")
print(X_scaled.shape)

# Menampilkan contoh hasil scaling
print("\nContoh hasil scaling:")
print(
    pd.DataFrame(
        X_scaled,
        columns=X.columns
    ).head()
)


# ============================================================
# K-MEANS UNTUK k = 2 SAMPAI 6
# ============================================================

hasil_k = []

model_kmeans = {}

for k in range(2, 7):

    # Membuat model K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    # Melatih model
    labels = kmeans.fit_predict(X_scaled)

    # Menghitung silhouette score
    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    # Menyimpan model
    model_kmeans[k] = kmeans

    # Menyimpan hasil
    hasil_k.append({
        "k": k,
        "inertia": kmeans.inertia_,
        "silhouette": silhouette
    })


# Mengubah hasil menjadi DataFrame
hasil_k_df = pd.DataFrame(hasil_k)

print("\nHasil pengujian k = 2 sampai 6:")
print(hasil_k_df)


# ============================================================
# MENENTUKAN k TERBAIK
# ============================================================

best_k = int(
    hasil_k_df.loc[
        hasil_k_df["silhouette"].idxmax(),
        "k"
    ]
)

print("\nK terbaik =", best_k)

print(
    "Silhouette terbaik =",
    round(
        hasil_k_df.loc[
            hasil_k_df["silhouette"].idxmax(),
            "silhouette"
        ],
        4
    )
)


# ============================================================
# VISUALISASI ELBOW METHOD
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    hasil_k_df["k"],
    hasil_k_df["inertia"],
    marker="o"
)

plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.xticks(range(2, 7))
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# VISUALISASI SILHOUETTE SCORE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    hasil_k_df["k"],
    hasil_k_df["silhouette"],
    marker="o"
)

plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score k = 2-6")

plt.xticks(range(2, 7))
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# 3. LATIH DAN TAFSIRKAN K-MEANS
# ============================================================

print("\n" + "=" * 60)
print("3. LATIH DAN TAFSIRKAN K-MEANS")
print("=" * 60)

# Mengambil model dengan k terbaik
kmeans_final = model_kmeans[best_k]

# Mendapatkan label cluster
cluster_kmeans = kmeans_final.labels_

# Membuat tabel hasil
hasil_kmeans = pd.DataFrame({
    "perfume": nama_perfume,
    "cluster": cluster_kmeans
})

print("\nHasil clustering:")
print(hasil_kmeans)


# ============================================================
# JUMLAH DATA SETIAP CLUSTER
# ============================================================

print("\nJumlah data setiap cluster:")

jumlah_cluster = (
    hasil_kmeans["cluster"]
    .value_counts()
    .sort_index()
)

print(jumlah_cluster)


# ============================================================
# PROFIL CLUSTER
# ============================================================

# Menambahkan cluster ke data asli
X_profil = X.copy()

X_profil["cluster"] = cluster_kmeans

# Menghitung rata-rata setiap fitur
profil_cluster = (
    X_profil
    .groupby("cluster")
    .mean()
)

print("\nRata-rata fitur setiap cluster:")
print(profil_cluster)


# ============================================================
# MENENTUKAN RATA-RATA KESELURUHAN FITUR
# ============================================================

rata_rata_cluster = {}

for cluster in sorted(
    np.unique(cluster_kmeans)
):

    nilai = X[
        cluster_kmeans == cluster
    ].to_numpy()

    rata_rata_cluster[cluster] = (
        nilai.mean()
    )


print("\nRata-rata keseluruhan fitur:")

for cluster, nilai in rata_rata_cluster.items():

    print(
        f"Cluster {cluster}: "
        f"{nilai:.2f}"
    )


# ============================================================
# MEMBERIKAN NAMA CLUSTER
# ============================================================

urutan_cluster = sorted(
    rata_rata_cluster,
    key=rata_rata_cluster.get
)

nama_cluster = {}

nama_cluster[
    urutan_cluster[0]
] = "Profil nilai fitur rendah"

nama_cluster[
    urutan_cluster[1]
] = "Profil nilai fitur menengah"

nama_cluster[
    urutan_cluster[2]
] = "Profil nilai fitur tinggi"


# Menambahkan nama cluster
hasil_kmeans["nama_cluster"] = (
    hasil_kmeans["cluster"]
    .map(nama_cluster)
)

print("\nHasil dengan nama cluster:")
print(hasil_kmeans)


# ============================================================
# USULAN STRATEGI
# ============================================================

print("\n" + "=" * 60)
print("USULAN STRATEGI")
print("=" * 60)

for cluster in sorted(nama_cluster):

    nama = nama_cluster[cluster]

    jumlah = (
        cluster_kmeans == cluster
    ).sum()

    anggota = list(
        nama_perfume[
            cluster_kmeans == cluster
        ]
    )

    print("\nCluster", cluster)
    print("Nama:", nama)
    print("Jumlah data:", jumlah)

    print("Anggota:")
    print(", ".join(anggota))

    if "tinggi" in nama:

        print(
            "Strategi: kelompok ini dapat "
            "dijadikan segmen khusus karena "
            "memiliki nilai fitur relatif tinggi."
        )

    elif "rendah" in nama:

        print(
            "Strategi: kelompok ini perlu "
            "dianalisis lebih lanjut karena "
            "jumlah datanya sedikit dan "
            "nilai fiturnya relatif rendah."
        )

    else:

        print(
            "Strategi: kelompok ini dapat "
            "dijadikan kelompok utama/standar "
            "karena berisi sebagian besar data."
        )


# ============================================================
# VISUALISASI HASIL K-MEANS
# ============================================================

plt.figure(figsize=(8, 5))

for cluster in sorted(
    np.unique(cluster_kmeans)
):

    mask = (
        cluster_kmeans == cluster
    )

    plt.scatter(
        X_scaled[mask, 0],
        X_scaled[mask, 1],
        label=f"Cluster {cluster}"
    )

plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

plt.title(
    f"Visualisasi K-Means (k={best_k})"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# 4. BANDINGKAN DENGAN DBSCAN
# ============================================================

print("\n" + "=" * 60)
print("4. PERBANDINGAN DENGAN DBSCAN")
print("=" * 60)


# ============================================================
# K-DISTANCE GRAPH
# ============================================================

neighbors = NearestNeighbors(
    n_neighbors=3
)

neighbors.fit(X_scaled)

distances, indices = (
    neighbors.kneighbors(X_scaled)
)

# Mengambil jarak tetangga ke-3
k_distance = np.sort(
    distances[:, -1]
)

plt.figure(figsize=(8, 5))

plt.plot(k_distance)

plt.xlabel("Data terurut")
plt.ylabel("Jarak ke tetangga ke-3")

plt.title(
    "K-Distance Graph untuk DBSCAN"
)

plt.grid(alpha=0.3)

plt.show()


# ============================================================
# MENJALANKAN DBSCAN
# ============================================================

eps = 1.75
min_samples = 3

dbscan = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

cluster_dbscan = (
    dbscan.fit_predict(X_scaled)
)


# Membuat tabel hasil DBSCAN
hasil_dbscan = pd.DataFrame({
    "perfume": nama_perfume,
    "cluster_dbscan": cluster_dbscan
})

print("\nParameter DBSCAN:")
print("eps =", eps)
print("min_samples =", min_samples)

print("\nHasil DBSCAN:")
print(hasil_dbscan)


# ============================================================
# JUMLAH CLUSTER DAN NOISE
# ============================================================

jumlah_cluster_dbscan = len(
    set(cluster_dbscan) - {-1}
)

jumlah_noise = (
    cluster_dbscan == -1
).sum()

print(
    "\nJumlah cluster DBSCAN:",
    jumlah_cluster_dbscan
)

print(
    "Jumlah noise:",
    jumlah_noise
)


# ============================================================
# SILHOUETTE DBSCAN
# ============================================================

bukan_noise = (
    cluster_dbscan != -1
)

if (
    jumlah_cluster_dbscan >= 2
    and bukan_noise.sum() > jumlah_cluster_dbscan
):

    silhouette_dbscan = silhouette_score(
        X_scaled[bukan_noise],
        cluster_dbscan[bukan_noise]
    )

    print(
        "Silhouette DBSCAN:",
        round(
            silhouette_dbscan,
            4
        )
    )


# ============================================================
# DATA NOISE
# ============================================================

print("\nData yang dianggap NOISE:")

noise_data = hasil_dbscan[
    hasil_dbscan["cluster_dbscan"] == -1
]

print(noise_data)


# ============================================================
# PERBANDINGAN K-MEANS DAN DBSCAN
# ============================================================

print("\n" + "=" * 60)
print("PERBANDINGAN K-MEANS DAN DBSCAN")
print("=" * 60)

print("\nK-MEANS")
print(
    "Jumlah cluster:",
    best_k
)

print(
    "Silhouette:",
    round(
        hasil_k_df.loc[
            hasil_k_df["k"] == best_k,
            "silhouette"
        ].iloc[0],
        4
    )
)

print("\nDBSCAN")
print(
    "Jumlah cluster:",
    jumlah_cluster_dbscan
)

print(
    "Jumlah noise:",
    jumlah_noise
)

if (
    jumlah_cluster_dbscan >= 2
    and bukan_noise.sum() > jumlah_cluster_dbscan
):

    print(
        "Silhouette:",
        round(
            silhouette_dbscan,
            4
        )
    )


# ============================================================
# KESIMPULAN
# ============================================================

print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)

print(
    f"""
Berdasarkan pengujian K-Means dengan k=2 sampai k=6,
nilai silhouette tertinggi diperoleh pada k={best_k}.

Dengan demikian, jumlah cluster yang dipilih adalah {best_k}.

K-Means membagi data perfume menjadi {best_k} kelompok
berdasarkan kemiripan karakteristik fitur.

Selanjutnya, hasil K-Means dibandingkan dengan DBSCAN.
DBSCAN menghasilkan {jumlah_cluster_dbscan} cluster
dan {jumlah_noise} data yang dianggap sebagai noise.

K-Means lebih sesuai jika seluruh data ingin dibagi
ke dalam sejumlah kelompok tertentu.

Sedangkan DBSCAN lebih sesuai jika analisis juga bertujuan
untuk menemukan data yang memiliki karakteristik berbeda
atau dianggap sebagai noise/outlier.
"""
)


# ============================================================
# MENYIMPAN HASIL KE EXCEL
# ============================================================

hasil_k_df.to_excel(
    "hasil_kmeans_k2_k6.xlsx",
    index=False
)

hasil_kmeans.to_excel(
    "hasil_cluster_kmeans.xlsx",
    index=False
)

hasil_dbscan.to_excel(
    "hasil_cluster_dbscan.xlsx",
    index=False
)

print("\nSemua hasil berhasil disimpan ke file Excel.")