#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# df = pd.read_csv("dataset/book_data.csv")

df = pd.read_excel(
    "dataset_cluster_added.xlsx",
)
df.fillna(value="", inplace=True)


df.drop_duplicates(subset=["book_title"], inplace=True)


df["genres"] = df["genres"].apply(lambda x: x.split("|"))

df["book_authors"] = df["book_authors"].apply(lambda x: x.split("|"))

df["book_pages"] = pd.to_numeric(
    df["book_pages"].str.replace(" pages", ""), errors="coerce"
)

df["book_pages"] = (
    df["book_pages"]
    .astype(str)
    .str.replace(" pages", "")
    .replace("", np.nan)
    .astype(float)
)


def recommend(selected_cluster, title, df):
    df2 = pd.read_excel(
        "dataset_cluster_added.xlsx",
    )
    df2.fillna(value="", inplace=True)

    df2.drop_duplicates(subset=["book_title"], inplace=True)

    df2["genres"] = df2["genres"].apply(lambda x: x.split("|"))

    df2["book_authors"] = df2["book_authors"].apply(lambda x: x.split("|"))

    df2["book_pages"] = pd.to_numeric(
        df2["book_pages"].str.replace(" pages", ""), errors="coerce"
    )

    df2["book_pages"] = (
        df2["book_pages"]
        .astype(str)
        .str.replace(" pages", "")
        .replace("", np.nan)
        .astype(float)
    )
    print("\n\n\t\tChecking true or not\n\n")
    print(df == df2)
    # filter DataFrame to include only rows with selected cluster number
    df = df.loc[df["Cluster Number"] == selected_cluster]
    df = df.reset_index(drop=True)
    vectorizer = TfidfVectorizer(stop_words="english")
    print(vectorizer)
    books_desc_vector = vectorizer.fit_transform(df["book_desc"].values.astype(str))

    # print(f"\nBook description vector shape is {books_desc_vector.shape}\n")

    all_in_vectors = vectorizer.get_feature_names_out()

    cos_similarities = cosine_similarity(books_desc_vector)
    # print(f"Cosine similarities shape {cos_similarities.shape}\n")

    df2 = df[["book_title", "book_authors", "book_desc"]]

    indices = pd.Series(df2.index, index=df["book_title"]).drop_duplicates()

    book_index = indices[title]
    similarity_scores = list(enumerate(cos_similarities[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_books = [i[0] for i in similarity_scores[:10]]
    recommended_books = df2.iloc[top_books]
    rec_books_list = recommended_books["book_title"].tolist()
    new_df = df[df["book_title"].isin(rec_books_list)]
    return new_df


# book_name = input("Enter any name of the books\n")
# selected_cluster = 3
# recs2 = recommend(selected_cluster, book_name, df)
# print(recs2)


'''
import string

df["name_feature"] = ["name_{}".format(x) for x in df["book_title"]]


def raw_text_to_feature(
    s,
    sep=" ",
    join_sep="x",
    to_include=string.ascii_lowercase,
    to_include2=string.ascii_uppercase,
):
    def filter_word(word):
        return "".join([c for c in word if (c in to_include or to_include2)])

    return join_sep.join([filter_word(word) for word in s.split(sep)])

df["name_feature"] = df["book_title"].apply(raw_text_to_feature)


df["corpus"] = pd.Series(
    df[["book_desc", "name_feature"]].fillna("").values.tolist()
).str.join(" ")
df["corpus"] = df["corpus"].fillna("")


# print(
# f'Corpus count -> {df["corpus"].count()}, Name feature count -> {df["name_feature"].count()}'
# )


vectorizer2 = TfidfVectorizer(stop_words="english")
books_desc_vector2 = vectorizer.fit_transform(df["corpus"].values.astype(str))
all_in_vectors2 = vectorizer.get_feature_names_out()

cos_similarities2 = cosine_similarity(books_desc_vector2)
print(type(cos_similarities2))
np.save("model_pickles/similarity_scores_genre_description.npy", cos_similarities2)
temp = np.load("model_pickles/similarity_scores_genre_description.npy")
print(temp == cos_similarities2)
"""

with open("model_pickles/similarity_scores_genre_description.pkl", "wb") as f:
    pickle.dump(cos_similarities2, f)
    print("Pickle dumped for genre description similarity scores")
"""


def give_recommendation_name_desc(title, cos_similarities=cos_similarities2):
    book_index = indices[title]
    similarity_scores = list(enumerate(cos_similarities[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    print(similarity_scores)
    top_books = [i[0] for i in similarity_scores[:10]]
    recommended_books = [df2.iloc[top_books]["book_title"].tolist()]
    print(recommended_books)
    # res = []
    # for i, book in enumerate(recommended_books[0]):
    #    res.append((book, similarity_scores[i][-1]))
    #    print(res[-1])
    return recommended_books


book_name = input("Enter any name of the books\n")
recs2 = give_recommendation_name_desc(book_name)
print(recs2)
'''
