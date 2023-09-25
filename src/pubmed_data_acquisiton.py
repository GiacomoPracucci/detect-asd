import os
from Bio import Entrez, Medline
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

EMAIL = "mail"
KEYWORDS = "(ASD OR autism) AND (eye-tracking OR eye_tracking OR scanpath OR scan-path)"
OUTPUT_PATH = 'pubmedgenerale.csv'


def fetch_pubmed_data(email, keywords):
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=keywords, retmax=10000)
    record = Entrez.read(handle)
    handle.close()
    idlist = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    df = pd.DataFrame(columns=["Titolo", "Autori", "Fonte", "Anno", "Abstract"])
    for i, record in enumerate(records):
        titolo = record.get("TI", "?")
        autori = ", ".join(record.get("AU", "?"))
        fonte = record.get("SO", "?")
        anno = record.get("DP", "?").split()[0] if record.get("DP", "?") != "?" else "?"
        abstract = record.get("AB", "?")
        df.loc[i] = [titolo, autori, fonte, anno, abstract]
    return df


def save_dataframe(df, output_path):
    df = df.sort_values("Anno").reset_index(drop=True).drop_duplicates(subset=['Abstract'])
    df.to_csv(output_path)


def visualize_word_cloud(df, column, title):
    text = ' '.join(df[column])
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()


def main():
    df = fetch_pubmed_data(EMAIL, KEYWORDS)
    save_dataframe(df, OUTPUT_PATH)
    visualize_word_cloud(df, 'Titolo', "Wordcloud dei titoli")
    visualize_word_cloud(df, 'Abstract', "Wordcloud degli abstract")


if __name__ == "__main__":
    main()
