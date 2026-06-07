# Eksperimen_SML_Bagus-Alfiyan-Yusuf

Repository untuk **Kriteria 1 - Experimentasi** pada proyek MSML (Membangun Sistem Machine Learning).

## Deskripsi

Proyek klasifikasi gambar hama cabai merah menggunakan CNN (TensorFlow/Keras). Dataset: [Red Chili Pepper Pests Dataset](https://www.kaggle.com/datasets/indraagustian/red-chili-pepper-pests-dataset) dari Kaggle.

## Struktur

```
├── preprocessing/
│   ├── Eksperimen_Bagus-Alfiyan-Yusuf.ipynb  # Notebook eksperimen
│   └── automate_Bagus-Alfiyan-Yusuf.py        # Script preprocessing otomatis
├── .github/workflows/preprocessing.yml         # GitHub Actions workflow
└── README.md
```

## Kelas Target

| ID | Nama | Deskripsi |
|----|------|-----------|
| 0 | MP (kutu daun) | Aphid |
| 1 | BT (kutu kebul) | Whitefly |
| 2 | T (thrips) | Thrips |
| 3 | C (ulat) | Caterpillar |

## Setup & Run

```bash
# Install dependencies
uv sync

# Run preprocessing
cd preprocessing
python automate_Bagus-Alfiyan-Yusuf.py --dataset-dir /path/to/dataset --output-dir red_chili_pepper_pests_preprocessed
```

## Dataset

Dataset dapat diunduh dari Kaggle:
```bash
kaggle datasets download -d indraagustian/red-chili-pepper-pests-dataset
```

## Author

- **Nama:** Bagus Alfiyan Yusuf
- **Dicoding Username:** fiyanz
