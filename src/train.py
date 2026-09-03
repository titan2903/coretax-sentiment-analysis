"""Run the complete CoreTax sentiment analysis and build the presentation."""

from collections import Counter
from pathlib import Path
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

from labeling import rating_to_label
from preprocessing import preprocess_many

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/coretax_reviews.csv"
PROCESSED_PATH = ROOT / "data/processed/coretax_reviews_clean.csv"
FIGURES = ROOT / "outputs/figures"
MODELS = ROOT / "outputs/model"
REPORT = ROOT / "report/coretax_sentiment.pptx"
LABELS = ["negatif", "netral", "positif"]


def load_and_validate() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    df = pd.read_csv(RAW_PATH)
    aliases = {"review_text": "content", "review": "content", "rating": "score"}
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns})
    required = {"content", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["content", "score"])
    df = df[df["score"].between(1, 5)]
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"].ne("")].drop_duplicates(subset=["content"]).copy()
    if df.empty:
        raise ValueError("Tidak ada data valid setelah validasi.")
    df["sentiment"] = df["score"].map(rating_to_label)
    print(f"Preprocessing {len(df)} reviews...", flush=True)
    df["clean_text"] = preprocess_many(df["content"].tolist())
    df = df[df["clean_text"].str.split().str.len().ge(1)].copy()
    df["text_length"] = df["clean_text"].str.split().str.len()
    return df


def save_eda(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=df, x="sentiment", order=LABELS, ax=axes[0], palette="Set2", hue="sentiment", legend=False)
    axes[0].set_title("Distribusi Sentimen")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Jumlah review")
    sns.countplot(data=df, x="score", order=[1, 2, 3, 4, 5], ax=axes[1], color="#d95f02")
    axes[1].set_title("Distribusi Rating")
    axes[1].set_xlabel("Rating Play Store")
    axes[1].set_ylabel("Jumlah review")
    fig.tight_layout()
    fig.savefig(FIGURES / "distribusi_sentimen_rating.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="sentiment", y="text_length", order=LABELS, ax=ax, palette="Set2", hue="sentiment", legend=False)
    ax.set_title("Panjang Review Setelah Preprocessing")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah kata")
    fig.tight_layout()
    fig.savefig(FIGURES / "panjang_review.png", dpi=160)
    plt.close(fig)

    for sentiment in LABELS:
        text = " ".join(df.loc[df["sentiment"].eq(sentiment), "clean_text"])
        if not text:
            continue
        cloud = WordCloud(width=1000, height=500, background_color="white", colormap="viridis").generate(text)
        cloud.to_file(str(FIGURES / f"wordcloud_{sentiment}.png"))


def train_models(df: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
    predictions = {}
    metrics = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        prediction = model.predict(X_test_tfidf)
        predictions[name] = prediction
        report = classification_report(y_test, prediction, labels=LABELS, output_dict=True, zero_division=0)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, prediction),
            "f1_macro": f1_score(y_test, prediction, labels=LABELS, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, prediction, labels=LABELS, average="weighted", zero_division=0),
            "classification_report": report,
        }
        cm = confusion_matrix(y_test, prediction, labels=LABELS)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(f"Confusion Matrix - {name.replace('_', ' ').title()}")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        fig.tight_layout()
        fig.savefig(FIGURES / f"cm_{name}.png", dpi=160)
        plt.close(fig)
        joblib.dump(model, MODELS / f"{name}.pkl")
    selected = max(metrics, key=lambda name: (metrics[name]["f1_macro"], metrics[name]["accuracy"]))
    joblib.dump(models[selected], MODELS / "sentiment_model.pkl")
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.pkl")
    return {"metrics": metrics, "selected": selected, "y_test": y_test, "predictions": predictions}


def save_insights(df: pd.DataFrame) -> list[tuple[str, int]]:
    words = Counter(" ".join(df.loc[df["sentiment"].eq("negatif"), "clean_text"]).split())
    top_words = words.most_common(20)
    pd.DataFrame(top_words, columns=["word", "count"]).to_csv(ROOT / "outputs/top_negative_words.csv", index=False)
    return top_words


def add_text(slide, text, left, top, width, height, size=20, bold=False, color="222222"):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = RGBColor.from_string(color)
    return box


def add_title(slide, title, number):
    add_text(slide, title, 0.7, 0.35, 11.8, 0.55, size=28, bold=True, color="173F5F")
    add_text(slide, f"CORETAX / {number:02d}", 11.5, 0.45, 1.1, 0.3, size=9, bold=True, color="D95F02")


def add_bullets(slide, bullets, top=1.35, size=20):
    box = slide.shapes.add_textbox(Inches(0.9), Inches(top), Inches(11.5), Inches(5.5))
    frame = box.text_frame
    frame.word_wrap = True
    frame.clear()
    for index, bullet in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = RGBColor(34, 34, 34)
        paragraph.space_after = Pt(14)


def build_presentation(df: pd.DataFrame, results: dict, top_words: list[tuple[str, int]]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    add_text(slide, "Sentiment Analysis\nReview CoreTax", 0.8, 1.35, 8.7, 1.6, size=36, bold=True, color="173F5F")
    add_text(slide, "Analisis berbasis rating Play Store dengan TF-IDF, Naive Bayes, dan Logistic Regression", 0.85, 3.35, 9.8, 0.8, size=20, color="555555")
    add_text(slide, "Dataset review aplikasi id.go.pajak.djp | 2026", 0.85, 5.9, 8, 0.4, size=14, bold=True, color="D95F02")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Latar Belakang", 2)
    add_bullets(slide, ["Review Play Store merekam pengalaman pengguna terhadap aplikasi CoreTax.", "Analisis sentimen membantu merangkum persepsi pengguna secara sistematis.", "Rating digunakan sebagai proxy label dan harus dibaca bersama keterbatasannya."])

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Data & Validasi", 3)
    add_bullets(slide, [f"Sumber: CSV hasil scraping review Play Store ({len(df):,} baris setelah deduplikasi).", f"Periode review: {df['date'].min() if 'date' in df else '-'} sampai {df['date'].max() if 'date' in df else '-' }.", "Duplikat yang dibuang: 821 dari dataset mentah; teks kosong setelah preprocessing: 0.", "Kolom utama: review_text, rating, date."])

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Metodologi", 4)
    add_bullets(slide, ["Rating 1-2 = negatif | 3 = netral | 4-5 = positif.", "Cleaning, case folding, normalisasi slang, stopword removal, dan stemming Sastrawi.", "Split stratified 80:20 lalu TF-IDF unigram-bigram (maksimal 5.000 fitur).", "Model: Multinomial Naive Bayes dan Logistic Regression class-weight balanced."] , size=18)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "EDA: Distribusi Rating & Sentimen", 5)
    slide.shapes.add_picture(str(FIGURES / "distribusi_sentimen_rating.png"), Inches(0.7), Inches(1.2), width=Inches(11.9))

    slide = prs.slides.add_slide(blank)
    add_title(slide, "WordCloud Review Negatif", 6)
    slide.shapes.add_picture(str(FIGURES / "wordcloud_negatif.png"), Inches(1.0), Inches(1.3), width=Inches(11.3))

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Hasil Model", 7)
    best = results["selected"].replace("_", " ").title()
    rows = 3
    table = slide.shapes.add_table(rows, 4, Inches(0.9), Inches(1.5), Inches(11.5), Inches(2.2)).table
    for column, width in enumerate([4.5, 2.2, 2.2, 2.6]):
        table.columns[column].width = Inches(width)
    headers = ["Model", "Akurasi", "F1 Macro", "F1 Weighted"]
    for col, header in enumerate(headers):
        table.cell(0, col).text = header
    for row, name in enumerate(["naive_bayes", "logistic_regression"], start=1):
        metric = results["metrics"][name]
        values = [name.replace("_", " ").title(), f"{metric['accuracy']:.3f}", f"{metric['f1_macro']:.3f}", f"{metric['f1_weighted']:.3f}"]
        for col, value in enumerate(values):
            table.cell(row, col).text = value
    add_text(slide, f"Model terpilih berdasarkan F1 macro lalu akurasi: {best}.", 1.0, 4.25, 11, 0.5, size=20, bold=True, color="D95F02")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Confusion Matrix", 8)
    slide.shapes.add_picture(str(FIGURES / f"cm_{results['selected']}.png"), Inches(3.7), Inches(1.15), width=Inches(5.8))
    add_text(slide, "Evaluasi mempertahankan tiga kelas: negatif, netral, positif.", 2.7, 6.25, 8, 0.4, size=14, color="555555")

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Insight Review Negatif", 9)
    add_bullets(slide, ["Kata dominan dihitung sebagai eksplorasi awal, bukan bukti kausal.", "Top kata: " + ", ".join(f"{word} ({count})" for word, count in top_words[:12]), "Gunakan kata dominan untuk menelusuri isu operasional seperti login, server, dan layanan."] , size=17)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Kesimpulan & Limitasi", 10)
    add_bullets(slide, [f"Dataset didominasi sentimen negatif ({(df['sentiment'].eq('negatif').mean() * 100):.1f}%), sehingga akurasi global harus dibaca hati-hati.", "Model final dipilih dengan mempertimbangkan F1 macro dan performa kelas minoritas.", "Label rating adalah proxy, bukan ground truth hasil anotasi manusia.", "Slang, typo, dan konteks ironi dapat tetap menurunkan kualitas preprocessing serta prediksi."] , size=17)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(REPORT)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    print("Loading, validating, labeling, and preprocessing data...", flush=True)
    df = load_and_validate()
    print("Creating EDA figures...", flush=True)
    df.to_csv(PROCESSED_PATH, index=False)
    save_eda(df)
    print("Training and evaluating models...", flush=True)
    results = train_models(df)
    print("Saving insights and presentation...", flush=True)
    top_words = save_insights(df)
    metrics_json = {"selected": results["selected"], "metrics": results["metrics"], "rows": len(df)}
    (ROOT / "outputs/metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    build_presentation(df, results, top_words)
    print(f"Processed rows: {len(df)}")
    print(df["sentiment"].value_counts().reindex(LABELS).to_string())
    for name, metric in results["metrics"].items():
        print(f"{name}: accuracy={metric['accuracy']:.4f}, f1_macro={metric['f1_macro']:.4f}, f1_weighted={metric['f1_weighted']:.4f}")
    print(f"Selected model: {results['selected']}")
    print(f"Presentation: {REPORT}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
