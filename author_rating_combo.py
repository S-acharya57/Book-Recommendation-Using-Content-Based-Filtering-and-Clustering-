"""
This is responsible for just suggesting books with same authors in the order of book_ratings.
A book can have multiple authors. Thus, it includes all the books having similar authors but in chronological
order of books based on book_rating column of the excel sheet.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset/book_data.csv")
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


# print(df.describe())
# print(df.info())
# print(df.info)

"""
authors = [author for authors in df["book_authors"] for author in authors]
unique_authors = list(set(authors))

le = LabelEncoder()

author_labels = le.fit_transform(unique_authors)
author_dict = dict(zip(unique_authors, author_labels))

# Create a new column in the dataframe with the integer labels for each author
df["author_labels"] = [
    list(map(lambda x: author_dict[x], authors)) for authors in df["book_authors"]
]

# print(df["author_labels"])

"""


def recommend(inp_book_title, df_all):
    authors = [author for authors in df_all["book_authors"] for author in authors]
    unique_authors = list(set(authors))

    le = LabelEncoder()

    author_labels = le.fit_transform(unique_authors)
    author_dict = dict(zip(unique_authors, author_labels))

    # Create a new column in the dataframe with the integer labels for each author
    df_all["author_labels"] = [
        list(map(lambda x: author_dict[x], authors))
        for authors in df_all["book_authors"]
    ]

    book_index = df_all.index[df_all["book_title"] == inp_book_title].tolist()[0]
    # print(f'Book index is {book_index}')

    book_authors = df_all.loc[book_index, "author_labels"]
    # print(f'Book authors are {book_authors}')

    # gives top rated books in order of the same authors
    recommended_books = df_all[
        df_all["author_labels"].apply(lambda x: set(x) == set(book_authors))
    ].sort_values("book_rating")
    # print(recommended_books)

    if book_index in recommended_books.index:
        recommended_books = recommended_books.drop(book_index)

    return recommended_books


# book_name = input("Enter the name of a book")
# books = recommend(book_name, df)
# print(books["book_title"])
