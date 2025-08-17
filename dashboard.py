import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#  Load model dan vectorizer utama
with open('model_forest.pkl', 'rb') as f:
    model_rf = pickle.load(f)

with open('model_bayes.pkl', 'rb') as f:
    model_nb = pickle.load(f)

with open('vectorizer_tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load model sentimen per aspek 
aspects = ["mudah", "puas", "efisiensi", "error"]
sentiment_models = {
    "Random Forest": {
        aspect: pickle.load(open(f"forest_sentimen_{aspect}.pkl", "rb"))
        for aspect in aspects
    },
    "Naive Bayes": {
        aspect: pickle.load(open(f"bayes_sentimen_{aspect}.pkl", "rb"))
        for aspect in aspects
    }
}

st.title("Dashboard Analisis Sentimen Berbasis Aspek Fasih")

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = None
if 'tfidf' not in st.session_state:
    st.session_state.tfidf = None
if 'prediction_aspect' not in st.session_state:
    st.session_state.prediction_aspect = None
if 'model_used' not in st.session_state:
    st.session_state.model_used = None

uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'content' not in df.columns:
        st.error("CSV harus memiliki kolom 'content'.")
        st.stop()

    st.write("Data Diupload:")
    st.dataframe(df.head())

    # === Preprocessing  ===
    if st.session_state.preprocessed is None:
        def simple_preprocess(text):
            import re
            from nltk.tokenize import word_tokenize
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

            stop_factory = StopWordRemoverFactory()
            stopwords = set(stop_factory.get_stop_words())
            stopwords.update(['untuk', 'kan', 'sih', 'kalo'])

            stemmer = StemmerFactory().create_stemmer()

            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in stopwords]
            tokens = [stemmer.stem(t) for t in tokens]

            return ' '.join(tokens)

        with st.spinner("Melakukan preprocessing..."):
            df['preprocessed'] = df['content'].apply(simple_preprocess)
            tfidf = vectorizer.transform(df['preprocessed'])

        st.session_state.df = df
        st.session_state.preprocessed = df['preprocessed']
        st.session_state.tfidf = tfidf
        st.success("Preprocessing selesai.")
    else:
        df = st.session_state.df
        tfidf = st.session_state.tfidf
        st.info("Preprocessing sudah selesai.")

    col1, col2 = st.columns(2)
    if col1.button("Naive Bayes"):
        with st.spinner("Prediksi aspek dengan Naive Bayes..."):
            pred = model_nb.predict(tfidf)
            for i, aspect in enumerate(aspects):
                df[f"Predicted_{aspect}"] = pred[:, i]
            st.session_state.prediction_aspect = pred
            st.session_state.model_used = "Naive Bayes"
        st.success("Prediksi aspek selesai.")

    if col2.button("Random Forest"):
        with st.spinner("Prediksi aspek dengan Random Forest..."):
            pred = model_rf.predict(tfidf)
            for i, aspect in enumerate(aspects):
                df[f"Predicted_{aspect}"] = pred[:, i]
            st.session_state.prediction_aspect = pred
            st.session_state.model_used = "Random Forest"
        st.success("Prediksi aspek selesai.")


if st.session_state.prediction_aspect is not None and st.session_state.df is not None:
    if st.button("Prediksi Sentimen & Tampilkan Grafik"):
        df = st.session_state.df
        pred = st.session_state.prediction_aspect
        model_used = st.session_state.model_used

        any_aspect_detected = (pred.sum(axis=1) > 0).any()

        if not any_aspect_detected:
            st.warning("Tidak ada aspek yang terdeteksi pada data. Sentimen tidak diprediksi.")
            
            for aspect in aspects:
                df[f"Sentimen_{aspect}"] = None
        else:
            with st.spinner("Melakukan prediksi sentimen..."):
                for i, aspect in enumerate(aspects):
                    detected_rows = pred[:, i] == 1
                    if detected_rows.any():
                        sent_model = sentiment_models[model_used][aspect]
                        df.loc[detected_rows, f"Sentimen_{aspect}"] = sent_model.predict(tfidf[detected_rows])
                    else:
                        df[f"Sentimen_{aspect}"] = ""

                        
                        if detected_rows.any():
                            sent_model = sentiment_models[model_used][aspect]
                            df.loc[detected_rows, f"Sentimen_{aspect}"] = sent_model.predict(tfidf[detected_rows])
            st.success("Prediksi sentimen selesai.")

            #Grafik
            st.subheader("Distribusi Sentimen Gabungan")

            aspek_list = []
            sentimen_list = []
            jumlah_list = []

            for aspect in aspects:
                counts = df[f"Sentimen_{aspect}"].value_counts(dropna=True)
                positif = counts.get(1, 0)
                negatif = counts.get(0, 0)
                aspek_list.extend([aspect, aspect])
                sentimen_list.extend(["Positif", "Negatif"])
                jumlah_list.extend([positif, negatif])

            df_plot = pd.DataFrame({
                "Aspek": aspek_list,
                "Sentimen": sentimen_list,
                "Jumlah": jumlah_list
            })

            df_pivot = df_plot.pivot(index="Aspek", columns="Sentimen", values="Jumlah").fillna(0)
            df_pivot = df_pivot.reindex(aspects)

            x = range(len(aspects))
            bar_width = 0.35

            fig, ax = plt.subplots()
            bars1 = ax.bar([i - bar_width/2 for i in x], df_pivot['Negatif'], width=bar_width, label='Negatif', color='red')
            bars2 = ax.bar([i + bar_width/2 for i in x], df_pivot['Positif'], width=bar_width, label='Positif', color='green')

            ax.set_xticks(x)
            ax.set_xticklabels(aspects)
            ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Sentimen per Aspek")
            ax.legend()

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom')

            st.pyplot(fig)

        
        st.download_button("Download Hasil Akhir", df.to_csv(index=False), file_name="hasil_akhir_sentimen.csv")