# Proyek Machine Learning Sistem Rekomendasi 

## Project Overview

Proyek ini bertujuan untuk membangun sistem rekomendasi anime yang cerdas dan relevan menggunakan dataset "Anime Recommendations Database". Dataset ini merupakan kumpulan data anime dari MyAnimeList yang berisi informasi lebih dari 12.000 judul anime. Proyek ini akan fokus pada pembuatan model rekomendasi anime menggunakan pendekatan _content-based filtering_.

Dalam era digital saat ini, dengan banyaknya pilihan anime yang tersedia di berbagai platform, pengguna seringkali kesulitan menemukan tontonan baru yang sesuai dengan preferensi mereka. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan anime yang sesuai dengan selera mereka, sehingga pengalaman menonton menjadi lebih personal dan menyenangkan.

Proyek ini akan menggunakan pendekatan _content-based filtering_ untuk membangun model yang mampu memberikan rekomendasi yang akurat dan efisien berdasarkan karakteristik atau konten dari anime itu sendiri, terutama genre.

### Latar Belakang

Di tengah ledakan konten anime yang tersedia melalui berbagai platform digital dan situs ulasan seperti MyAnimeList, pengguna sering dihadapkan pada pilihan yang sangat banyak. Hal ini dapat menyebabkan _information overload_, di mana pengguna kesulitan menentukan anime mana yang akan mereka nikmati selanjutnya. Tanpa mekanisme penyaringan otomatis, pengguna bisa menghabiskan banyak waktu mencari atau bahkan kehilangan minat. Oleh karena itu, sistem rekomendasi menjadi alat yang krusial untuk meningkatkan pengalaman pengguna dan membantu mereka menemukan konten yang relevan.

Pendekatan _content-based filtering_ adalah salah satu metode yang umum digunakan, di mana rekomendasi dibuat berdasarkan kesamaan atribut atau konten antar item. Dalam konteks anime, atribut ini bisa berupa genre, tipe, studio, atau kata kunci lainnya. Metode ini efektif ketika metadata anime tersedia secara detail. Keterbatasan utamanya adalah kurangnya kemampuan untuk merekomendasikan item di luar preferensi eksplisit pengguna (serendipity) dan masalah _cold start_ untuk item baru yang belum memiliki cukup deskripsi. Meskipun demikian, untuk dataset dengan informasi genre yang kaya seperti yang digunakan, _content-based filtering_ dapat memberikan rekomendasi yang solid dan mudah diinterpretasi.

## Business Understanding

Pengembangan sistem rekomendasi anime bertujuan untuk meningkatkan pengalaman pengguna dengan menyediakan saran anime yang lebih personal dan akurat, sekaligus membantu platform atau layanan terkait anime dalam meningkatkan keterlibatan pengguna.

### Problem Statements

- Bagaimana cara memahami dan memperoleh informasi mengenai data anime yang digunakan dalam pembuatan model sistem rekomendasi?
- Bagaimana cara membangun model sistem rekomendasi anime dengan menggunakan pendekatan _content-based filtering_ berdasarkan genre?
- Bagaimana cara menilai kinerja model sistem rekomendasi yang telah dikembangkan?

### Goals

- Melakukan eksplorasi data awal (_Exploratory Data Analysis / EDA_) dan visualisasi untuk memahami struktur serta karakteristik dataset anime.
- Membangun sistem rekomendasi anime menggunakan pendekatan _content-based filtering_ dengan memanfaatkan fitur genre.
- Mengevaluasi kinerja model sistem rekomendasi yang telah dibuat menggunakan metrik presisi.

### Solution Approach

Untuk memecahkan rumusan masalah dan mencapai tujuan yang telah ditentukan, pendekatan solusi yang sistematis dan terstruktur adalah sebagai berikut:

- **Melakukan eksplorasi data**:
  Langkah awal adalah memahami dataset yang digunakan (`anime.csv` dan `rating.csv`) melalui _exploratory data analysis_. Ini mencakup analisis jumlah baris dan kolom, tipe data, distribusi nilai (tipe anime, genre, rating), serta visualisasi grafik untuk mengidentifikasi pola atau anomali.
- **Mempersiapkan data (_Data Preparation_)**:
  Proses ini meliputi:
  - Penanganan nilai kosong (_missing values_).
  - Penghapusan data duplikat.
  - Pemrosesan teks pada kolom genre (menghilangkan koma, mengubah frasa multi-kata menjadi token tunggal).
- **Membangun sistem rekomendasi berbasis konten (_Content-Based Filtering_)**:
  - Transformasi data genre menggunakan TF-IDF (_Term Frequency-Inverse Document Frequency_) untuk membuat representasi numerik dari genre.
  - Menghitung kesamaan (_cosine similarity_) antar anime berdasarkan vektor TF-IDF genre mereka.
  - Pembuatan fungsi rekomendasi yang mengambil judul anime sebagai input dan mengembalikan daftar anime yang paling mirip.
- **Evaluasi kinerja model**:
  - Untuk model _content-based filtering_, digunakan metrik evaluasi _Precision@K_ untuk mengukur tingkat keakuratan dari sejumlah K rekomendasi teratas.

## Data Understanding

Dataset [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) digunakan dalam proyek ini adalah "Anime Recommendations Database", yang merupakan kumpulan data anime dari MyAnimeList. Dataset ini umumnya terdiri dari dua file utama:

- `anime.csv`: Berisi informasi detail tentang setiap anime.
- `rating.csv`: Berisi rating yang diberikan oleh pengguna terhadap anime.

Untuk proyek ini, fokus utama adalah pada `anime.csv` untuk membangun model _content-based filtering_ berdasarkan genre. File `rating.csv` juga dimuat untuk analisis eksplorasi data awal.

### Struktur Variabel Dataset

#### File: `anime.csv`

| Kolom      | Deskripsi                                                               |
| :--------- | :---------------------------------------------------------------------- |
| `anime_id` | ID unik untuk setiap anime.                                             |
| `name`     | Judul anime.                                                            |
| `genre`    | Kategori genre anime (misalnya: Action, Comedy, Drama, Fantasy).        |
| `type`     | Jenis anime (TV, Movie, OVA, dll).                                      |
| `episodes` | Jumlah episode.                                                         |
| `rating`   | Rata-rata rating yang diberikan pengguna.                               |
| `members`  | Jumlah pengguna yang telah memberi rating atau menonton anime tersebut. |

#### File: `rating.csv`

| Kolom      | Deskripsi                                                                                            |
| :--------- | :--------------------------------------------------------------------------------------------------- |
| `user_id`  | ID unik pengguna.                                                                                    |
| `anime_id` | ID anime yang diberi rating.                                                                         |
| `rating`   | Nilai rating yang diberikan pengguna terhadap anime (skala 1–10, atau -1 jika tidak memberi rating). |

### Exploratory Data Analysis (EDA)

Proses awal dalam analisis data yang bertujuan untuk memahami struktur, karakteristik, pola, dan hubungan dalam data sebelum dilakukan pemodelan atau analisis statistik lebih lanjut.

- Melihat info pada dataframe anime

```
anime_df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12294 entries, 0 to 12293
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   anime_id  12294 non-null  int64
 1   name      12294 non-null  object
 2   genre     12232 non-null  object
 3   type      12269 non-null  object
 4   episodes  12294 non-null  object
 5   rating    12064 non-null  float64
 6   members   12294 non-null  int64
dtypes: float64(1), int64(2), object(4)
memory usage: 672.5+ KB
```

- Melihat info pada dataframe rating

```
rating_df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7813737 entries, 0 to 7813736
Data columns (total 3 columns):
 #   Column    Dtype
---  ------    -----
 0   user_id   int64
 1   anime_id  int64
 2   rating    int64
dtypes: int64(3)
memory usage: 178.8 MB
```

- Melihat 5 data pertama pada dataframe anime

```
anime_df.head()
```

```
anime_id	name	genre	type	episodes	rating	members
32281	Kimi no Na wa.	Drama, Romance, School, Supernatural	Movie	1	9.37	200630
5114	Fullmetal Alchemist: Brotherhood	Action, Adventure, Drama, Fantasy, Magic, Mili...	TV	64	9.26	793665
28977	Gintama°	Action, Comedy, Historical, Parody, Samurai, S...	TV	51	9.25	114262
9253	Steins;Gate	Sci-Fi, Thriller	TV	24	9.17	673572
9969	Gintama'	Action, Comedy, Historical, Parody, Samurai, S...	TV	51	9.16	151266
```

- Melihat tipe dan beberapa sample genre anime

```
print("Tipe anime:", anime_df['type'].value_counts())
print("\nGenre sample:", anime_df['genre'].dropna().iloc[0])
```

```
Tipe anime: type
TV         3787
OVA        3311
Movie      2348
Special    1676
ONA         659
Music       488
Name: count, dtype: int64

Genre sample: Drama, Romance, School, Supernatural
```

- Melihat jumlah baris dan kolom pada dataframe anime dan rating

```
print("Jumlah baris dan kolom di anime dataframe:", anime_df.shape)

print("Jumlah baris dan kolom di rating dataframe:", rating_df.shape)
```

```
Jumlah baris dan kolom di anime dataframe: (12294, 7)
Jumlah baris dan kolom di rating dataframe: (7813737, 3)
```

- Melihat missing values pada dataframe anime dan rating

```
print("\nMissing values per fitur in anime dataframe:")
print(anime_df.isnull().sum())

print("\nMissing values per fitur in rating dataframe:")
print(rating_df.isnull().sum())
```

```

Missing values per fitur in anime dataframe:
anime_id      0
name          0
genre        62
type         25
episodes      0
rating      230
members       0
dtype: int64

Missing values per fitur in rating dataframe:
user_id     0
anime_id    0
rating      0
dtype: int64
```

Setelah dilihat, ternyata ada beberapa data pada dataframe anime yang memiliki missing value seperti genre, type, dan rating

- Melihat data duplikat pada dataframe anime dan rating

```
duplicates = anime_df.duplicated()
duplicate_count = duplicates.sum()

rating_duplicates = rating_df.duplicated()
rating_duplicate_count = rating_duplicates.sum()

print(f"Number of duplicate rows in anime dataframe: {duplicate_count}")

print(f"Number of duplicate rows in rating dataframe: {rating_duplicate_count}")
```

```
Number of duplicate rows in anime dataframe: 0
Number of duplicate rows in rating dataframe: 1
```

Setelah dilihat, ternyata ada 1 data yang duplikat pada dataframe rating

### Visualisasi Data

- Analisis distribusi tipe penayangan anime.

![Image](https://github.com/user-attachments/assets/1c0c917e-efb5-46b2-8082-9cb0737b9bca)

Grafik menunjukkan bahwa anime tipe TV merupakan yang paling banyak diproduksi, disusul oleh OVA dan Movie. Sementara itu, tipe seperti Special, ONA, dan Music jumlahnya jauh lebih sedikit. Hal ini menunjukkan bahwa format TV masih menjadi media utama dalam distribusi anime, sedangkan tipe lain bersifat pelengkap atau lebih spesifik.

- Distribusi Genre Anime

![Image](https://github.com/user-attachments/assets/fc48f727-2a37-4bba-be2b-86ab120de6c2)

Grafik menunjukkan bahwa genre anime paling populer adalah Comedy, diikuti oleh Action, Adventure, dan Fantasy. Genre seperti Yaoi, Yuri, dan Josei memiliki jumlah anime yang jauh lebih sedikit. Ini mengindikasikan bahwa preferensi industri dan penonton cenderung fokus pada genre yang ringan dan penuh aksi, sementara genre-genre yang lebih spesifik atau bertarget niche memiliki representasi yang lebih kecil.

- Distribusi Rating Anime

![Image](https://github.com/user-attachments/assets/acdb6cc1-1c60-439d-85ab-cd4e3611052f)

Grafik distribusi rating anime menunjukkan pola yang menyerupai kurva normal, dengan sebagian besar anime memiliki rating di kisaran 6 hingga 8. Sangat sedikit anime yang mendapatkan rating di bawah 4 atau di atas 9.

- Distribusi Rating Pengguna

![Image](https://github.com/user-attachments/assets/a76cb9e5-3f01-47f1-9828-6a2cdcd9d466)

Grafik menunjukkan kebanyakkan user menilai anime dengan score 8. Selain itu, banyak juga user yang belum menilai anime yang ditonton

- Top 10 Anime Berdasarkan Jumlah Rating Pengguna

![Image](https://github.com/user-attachments/assets/0e5a0bde-6bb3-4738-a4e4-9d53a516c325)

Grafik ini menunjukkan bahwa "Death Note" adalah anime dengan jumlah rating terbanyak secara signifikan, diikuti oleh "Sword Art Online" dan "Shingeki no Kyojin". Kehadiran dua seri dari "Code Geass" (original dan R2) serta "Fullmetal Alchemist" (original dan Brotherhood) dalam daftar ini menunjukkan popularitas dan basis penggemar yang kuat untuk franchise tersebut.

## Data Preparation

Pada tahap ini, data preparation adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model rekomendasi. Data mentah sering kali mengandung nilai kosong, duplikasi yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar modeling berjalan optimal.

- Menghapus Missing Value

```
anime_df = anime_df.dropna(subset=['rating'])
anime_df['genre'] = anime_df['genre'].fillna('')
print(anime_df.isnull().sum())
```

Output menunjukkan tidak ada lagi missing value yang signifikan untuk kolom yang akan digunakan.

- Menghilangkan Data Duplikat

```
anime_df = anime_df.drop_duplicates(subset='anime_id')
```

Data duplikat berdasarkan anime_id di anime_df diperiksa dan dihapus (meskipun pada kasus ini tidak ditemukan duplikat setelah filtering awal). Data duplikat pada rating_df juga akan diabaikan karena fokus utama adalah anime_df untuk content-based.

- Menghilangkan koma pada genre

```
anime_df['genre'] = anime_df['genre'].str.replace(',', ' ')
```

Untuk memudahkan analisis TF-IDF, karakter koma pada kolom genre diganti dengan spasi.

- Mengubah frasa multi-kata spesifik yang ingin dipertahankan sebagai satu token

```
anime_df['genre'] = anime_df['genre'].str.replace(r'slice of life', 'slice_of_life', regex=True)
anime_df['genre'] = anime_df['genre'].str.replace(r'sci fi', 'sci_fi', regex=True)
anime_df['genre'] = anime_df['genre'].str.replace(r'martial arts', 'martial_arts', regex=True)
anime_df['genre'] = anime_df['genre'].str.replace(r'super power', 'super_power', regex=True)
```

Frasa multi-kata spesifik (seperti "slice of life") diubah menjadi satu token (misalnya, "slice_of_life").

- TF-IDF Transformation

TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah teks genre anime menjadi representasi numerik (vektor) yang dapat diproses oleh model.

```
tfidf = TfidfVectorizer() #
tfidf_matrix = tfidf.fit_transform(anime_df['genre']) #
```

Hasilnya adalah matriks TF-IDF di mana setiap baris mewakili anime dan setiap kolom mewakili skor TF-IDF untuk setiap genre unik.

## Modeling

### Sistem Rekomendasi - Content Based Filtering

Pendekatan ini merekomendasikan item berdasarkan kesamaan karakteristik item tersebut. Dalam kasus ini, genre anime akan digunakan sebagai dasar kemiripan.

**Kelebihan:**

- Tidak memerlukan data dari pengguna lain.
- Dapat merekomendasikan item yang baru dan kurang populer jika memiliki deskripsi fitur yang baik.
- Rekomendasi bersifat transparan dan mudah dijelaskan (berdasarkan kesamaan genre).

**Kekurangan:**

- Fitur item harus diekstrak dengan baik.
- Cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, kurang variatif (overspecialization).
- Sulit menangani cold start untuk pengguna baru yang belum memiliki histori.

- Cosine Similarity

Cosine similarity digunakan untuk mengukur kesamaan antara vektor TF-IDF dari setiap pasang anime. Nilai kesamaan berkisar antara 0 (tidak mirip) hingga 1 (sangat mirip). Cara kerja cosine similarity sebagai berikut.

$$ \text{cosine similarity}(A, B) = \frac{A \cdot B}{|A| |B|} = \frac{\sum*{i=1}^{n} A_i B_i}{\sqrt{\sum*{i=1}^{n} A*i^2} \times \sqrt{\sum*{i=1}^{n} B_i^2}} $$

Rumus cosine similarity mengukur kesamaan antara dua vektor, misalnya vektor A dan vektor B, dengan menghitung kosinus sudut di antara keduanya. Perhitungan ini dimulai dengan dot product (produk skalar) dari kedua vektor, yang dinotasikan sebagai A ⋅ B. Ini diperoleh dengan mengalikan setiap elemen yang bersesuaian dari vektor A dan B, kemudian menjumlahkan semua hasil perkalian tersebut.

Selanjutnya, kita memerlukan norma atau magnitudo (panjang) dari masing-masing vektor. Norma vektor A, dinotasikan sebagai ∥A∥ , dihitung dengan mengakar kuadratkan jumlah dari kuadrat setiap elemen dalam vektor A. Proses yang sama berlaku untuk menghitung norma vektor B, ∥B∥​.

Akhirnya, cosine similarity didapatkan dengan membagi hasil dot product kedua vektor (A⋅B) dengan hasil perkalian norma kedua vektor tersebut (∥A∥∥B∥). Nilai yang dihasilkan berkisar antara -1 dan 1, di mana 1 menunjukkan kesamaan sempurna (arah vektor sama), 0 menunjukkan tidak ada kesamaan (vektor ortogonal), dan -1 menunjukkan vektor berlawanan arah. Dalam konteks TF-IDF di mana nilai frekuensi tidak negatif, hasilnya akan berada di antara 0 dan 1, dengan nilai yang semakin mendekati 1 menandakan kemiripan yang lebih tinggi antar item yang dibandingkan.

**Interpretasi Hasil:**

- 1: Menunjukkan bahwa kedua vektor memiliki orientasi yang sama (sudut 0 derajat). Dalam konteks data, ini berarti item-item tersebut sangat mirip.

- 0: Menunjukkan bahwa kedua vektor ortogonal (sudut 90 derajat), yang berarti tidak ada kesamaan.

- 1: Menunjukkan bahwa kedua vektor memiliki orientasi yang berlawanan (sudut 180 derajat).

```
cosine_sim = cosine_similarity(tfidf_matrix)
```

- Mapping Name To Index

Sebuah pandas.Series dibuat untuk memetakan nama anime ke indeksnya dalam DataFrame, memudahkan pencarian.

```
indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()
```

- Fungsi Sistem Rekomendasi

Fungsi `get_recommendations` dibuat untuk memberikan rekomendasi anime berdasarkan judul input dan matriks cosine similarity.

```
def get_recommendations(title, cosine_sim=cosine_sim, top_k=5):
    if title not in indices:
        return f"'{title}' tidak ditemukan."

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_k+1]
    anime_indices = [i[0] for i in sim_scores]

    return anime_df[['name', 'genre', 'type']].iloc[anime_indices].reset_index(drop=True)
```

- Menjalankan Sistem Rekomendasi

```
print(get_recommendations("Bakemonogatari"))
```

**Output:**

```
name  \
0  Monogatari Series: Second Season
1    Kizumonogatari I: Tekketsu-hen
2                    Vampire Holmes
3   Kizumonogatari II: Nekketsu-hen
4             Vampire Knight Guilty

                                               genre   type
0    Comedy  Mystery  Romance  Supernatural  Vampire     TV
1                     Mystery  Supernatural  Vampire  Movie
2             Comedy  Mystery  Supernatural  Vampire     TV
3             Action  Mystery  Supernatural  Vampire  Movie
4  Drama  Mystery  Romance  Shoujo  Supernatural ...     TV
```

Sistem berhasil merekomendasikan anime dengan genre yang mirip (Mystery, Supernatural, Vampire).

## Evaluation

Model akan diuji menggunakan metrik Precision@K. Precision mengukur seberapa banyak dari K item yang direkomendasikan benar-benar relevan dengan item input.

### Evaluasi Model Content-Based Filtering (Metrik: Precision@K)

Precision@K mengukur proporsi item yang relevan di antara K item teratas yang direkomendasikan.

Relevansi ditentukan berdasarkan kesamaan genre. Sebuah anime dianggap relevan jika memiliki setidaknya threshold genre yang sama dengan anime input.

**Cara Kerja**

1.  Mengambil genre dari anime input.
2.  Menentukan threshold jumlah genre yang harus sama agar dianggap relevan.
3.  Mendapatkan K rekomendasi teratas.
4.  Untuk setiap anime yang direkomendasikan, hitung jumlah genre yang sama dengan anime input.
5.  Jika jumlah genre yang sama ≥ threshold, anime tersebut dianggap relevan.
6.  Hitung Precision@K.

```
def assess_recommendations(query_title, df, similarity_matrix, k=5):
    if query_title not in indices:
        return f"Judul '{query_title}' tidak tersedia dalam dataset."

    query_index = indices[query_title]
    base_genre = df.loc[query_index, 'genre']
    genre_set = set(base_genre.lower().split()) if pd.notna(base_genre) else set()

    threshold = 2 if len(genre_set) >= 3 else 1

    similarity_rank = sorted(
        [(i, score) for i, score in enumerate(similarity_matrix[query_index]) if i != query_index],
        key=lambda x: x[1],
        reverse=True
    )

    top_matches = [i for i, _ in similarity_rank[:k]]
    results_df = df.iloc[top_matches].copy()  # Copy untuk keamanan

    relevancy_flags = []
    for genre_string in results_df['genre']:
        comparison_set = set(str(genre_string).lower().split()) if pd.notna(genre_string) else set()
        overlap = genre_set.intersection(comparison_set)
        relevancy_flags.append(len(overlap) >= threshold)

    score = sum(relevancy_flags)
    precision_score = score / k

    relevant_titles = results_df.iloc[[i for i, rel in enumerate(relevancy_flags) if rel]]['name'].tolist()
    irrelevant_titles = results_df.iloc[[i for i, rel in enumerate(relevancy_flags) if not rel]]['name'].tolist()

    return {
        'Title': query_title,
        'Top K': k,
        'Matches Found': k,
        'Matches Relevant': score,
        'Matches Irrelevant': k - score,
        'Precision@K': precision_score,
        'Relevant Titles': relevant_titles,
        'Irrelevant Titles': irrelevant_titles
    }
```

```
result = assess_recommendations("Bakemonogatari", anime_df, cosine_sim)
print(result)
```

**Output Evaluasi:**

```
{'Title': 'Bakemonogatari', 'Top K': 5, 'Matches Found': 5, 'Matches Relevant': 5, 'Matches Irrelevant': 0, 'Precision@K': 1.0, 'Relevant Titles': ['Monogatari Series: Second Season', 'Kizumonogatari I: Tekketsu-hen', 'Vampire Holmes', 'Kizumonogatari II: Nekketsu-hen', 'Vampire Knight Guilty'], 'Irrelevant Titles': []}
```

**Interpretasi Hasil**

Untuk anime "Bakemonogatari" (genre: Mystery, Romance, Supernatural, Vampire), kelima rekomendasi teratas memiliki kesamaan genre yang signifikan (setidaknya 2 genre yang sama, karena "Bakemonogatari" memiliki >= 3 genre). Dengan demikian, Precision@5 adalah 1.0, yang menunjukkan bahwa semua rekomendasi dianggap relevan oleh metrik evaluasi ini.

## Hubungan dengan Business Understanding & Mengatasi Problem Statements

- Proyek ini berhasil menjawab seluruh _problem statements_ yang telah ditetapkan. Melalui proses eksplorasi data (EDA), ditemukan karakteristik penting dari dataset anime seperti distribusi genre yang tidak merata dengan genre "Comedy" dan "Action" sebagai yang paling dominan, serta pola rating pengguna yang cenderung memberikan skor tinggi (terutama skor 8). Ini menjawab _problem statement_ pertama mengenai pemahaman dan perolehan informasi dari data yang digunakan.

- Proses _data preparation_ dilakukan secara menyeluruh dengan menangani _missing values_ pada kolom `rating` dan `genre`, serta melakukan pemrosesan teks pada kolom `genre` untuk menghilangkan koma dan mengubah frasa multi-kata menjadi token tunggal yang konsisten. Hal ini menjawab _problem statement_ kedua mengenai cara membangun model _content-based filtering_, karena persiapan data genre yang baik adalah kunci untuk representasi fitur yang akurat menggunakan TF-IDF.

- Dalam menjawab _problem statement_ ketiga mengenai penilaian kinerja model, sistem rekomendasi _content-based filtering_ yang dibangun dievaluasi menggunakan metrik _Precision@K_. Hasil evaluasi untuk contoh anime "Bakemonogatari" menunjukkan _Precision@5_ sebesar 1.0, menandakan bahwa kelima rekomendasi yang diberikan sangat relevan berdasarkan kesamaan genre. Ini menunjukkan kemampuan model dalam memberikan rekomendasi yang akurat sesuai dengan konten anime.

## Solution Statements

Beberapa langkah yang dijabarkan dalam _solution statements_ terbukti memberikan dampak signifikan terhadap kualitas data dan performa model:

- **EDA dan Visualisasi:**
  Analisis univariat pada tipe anime, genre, dan rating (baik rating anime maupun rating pengguna) membantu mengungkap pola penting dalam data. Visualisasi distribusi genre menunjukkan genre mana yang paling populer dan mana yang kurang terwakili, sementara distribusi rating anime dan pengguna memberikan gambaran tentang bagaimana anime umumnya dinilai. Analisis Top 10 anime berdasarkan jumlah rating juga memberikan insight tentang anime mana yang paling banyak berinteraksi dengan pengguna.

- **Data Preparation yang Fokus pada Genre:**
  Penanganan _missing values_ terutama pada kolom `rating` (dengan menghapus baris) dan `genre` (dengan mengisi string kosong) memastikan tidak ada data yang hilang untuk fitur utama. Pemrosesan teks pada kolom `genre`, seperti penghilangan koma dan standardisasi frasa multi-kata (misalnya, 'slice of life' menjadi 'slice_of_life'), sangat krusial untuk meningkatkan kualitas vektorisasi TF-IDF. Langkah ini memastikan bahwa genre yang sama namun ditulis berbeda tetap dihitung sebagai satu fitur yang sama, yang secara langsung meningkatkan akurasi perhitungan _cosine similarity_.

- **Pemodelan _Content-Based Filtering_ dengan TF-IDF dan Cosine Similarity:**
  Implementasi TF-IDF berhasil mengubah data tekstual genre menjadi representasi numerik yang dapat diukur kesamaannya. Penggunaan _cosine similarity_ kemudian secara efektif menghitung kemiripan antar anime berdasarkan vektor genre tersebut. Fungsi rekomendasi yang dibangun di atas matriks kesamaan ini mampu memberikan daftar anime yang relevan berdasarkan judul input.

- **Evaluasi dengan Precision@K:**
  Penggunaan metrik _Precision@K_ memberikan penilaian yang jelas dan terukur terhadap relevansi rekomendasi yang dihasilkan oleh model _content-based filtering_. Dengan menetapkan _threshold_ kesamaan genre untuk menentukan relevansi, evaluasi menunjukkan bahwa model mampu memberikan rekomendasi yang sangat tepat (Precision@5 = 1.0 untuk "Bakemonogatari"), yang berarti semua rekomendasi yang diberikan memiliki kesamaan genre yang signifikan dengan anime input.

### Author 
Nama : Cahya Abdurrahman

Email : cahyaabd@upi.edu
