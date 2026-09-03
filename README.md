# CoreTax Sentiment Analysis

Analisis sentimen review Play Store aplikasi CoreTax (`id.go.pajak.djp`) menggunakan label proxy dari rating, preprocessing Bahasa Indonesia, TF-IDF, Naive Bayes, dan Logistic Regression. Fungsi preprocessing individual menyediakan stemming Sastrawi; runner corpus memakai cleaning dan stopword removal yang stabil karena stemming Sastrawi pada dataset ini tidak selesai dalam waktu operasional.

## Menjalankan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Runner menghasilkan:

- `data/processed/coretax_reviews_clean.csv`: data tervalidasi, deduplikasi, berlabel, dan telah dipreprocess
- `outputs/figures/`: grafik distribusi, panjang teks, wordcloud, dan confusion matrix
- `outputs/model/`: kedua model, model terpilih, serta TF-IDF vectorizer
- `outputs/metrics.json`: metrik akurasi, F1 macro, F1 weighted, dan classification report
- `outputs/top_negative_words.csv`: kata dominan pada review negatif
- `report/coretax_sentiment.pptx`: delivery 10 slide

Label rating adalah proxy, bukan ground truth anotasi manusia. Dataset juga sangat tidak seimbang, sehingga F1 macro dan metrik per kelas perlu dibaca bersama akurasi.
