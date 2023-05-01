import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend(book_title_choice):
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

    mlb = MultiLabelBinarizer()

    genres_matrix = mlb.fit_transform(df["genres"])
    # print(genres_matrix.shape)

    kmeans_temp = KMeans(n_clusters=57, random_state=35).fit(genres_matrix)

    # print(kmeans_temp.labels_)

    cluster_counts = pd.Series(kmeans_temp.labels_).value_counts()
    # print(f"E.g. Number of books under cluster 52 is {cluster_counts.iloc[52]}")

    cluster_dict = {}
    for i, label in enumerate(kmeans_temp.labels_):
        if label not in cluster_dict:
            cluster_dict[label] = []
        genre_indices = np.where(genres_matrix[i] == 1)[0]
        genres = mlb.classes_[genre_indices]
        cluster_dict[label].extend(genres)
        cluster_dict[label] = list(set(cluster_dict[label]))

    # print(f"Books present in cluster 52 is :\n{cluster_dict[3]}")

    for i in range(kmeans_temp.n_clusters):
        cluster_books = df.iloc[np.where(kmeans_temp.labels_ == i)[0]][
            "book_title"
        ].tolist()
        # print(f"Listing 5 books of each cluster to have an idea for each cluster \n\n")
        # print(f"Cluster {i}: \n{', '.join(cluster_books[:5])}\n\n")
    df.reset_index(inplace=True)
    from sklearn.metrics.pairwise import cosine_similarity

    liked_book = book_title_choice

    # Identifying the cluster where the entered book falls into, for getting which are the closest ones
    liked_book_genre = genres_matrix[df.index[df["book_title"] == liked_book]]
    liked_book_genre = liked_book_genre.ravel()
    cluster_label = kmeans_temp.predict(liked_book_genre.reshape(1, -1))[0]

    # All the books that are present in the cluster is found out as
    books_in_cluster = df.loc[kmeans_temp.labels_ == cluster_label]["book_title"]

    # removing the liked book from the cluster first
    books_in_cluster = books_in_cluster[books_in_cluster != liked_book]

    temp = genres_matrix[df.index.isin(books_in_cluster)]
    # print(f"Genre matrix of liked book is\n {liked_book_genre}")
    # print(f"Cluster label of the book is {cluster_label}")

    # books_in_cluster = df.loc[kmeans_temp.labels_ == cluster_label]['book_title']
    # df.reset_index(inplace =True)
    similarities = []
    for book in books_in_cluster:
        book_genre = genres_matrix[df.index[df["book_title"] == book]]
        # print(book_genre.shape, liked_book_genre.shape)
        similarity = cosine_similarity(
            liked_book_genre.reshape(1, -1), book_genre.reshape(1, -1)
        )[0][0]
        similarities.append(similarity)
    # print(similarities)

    # Creating a new df of book titles and their corresponding similarity scores for the recommendation
    # similar to df but only has books and similarities
    similarities_df = pd.DataFrame(
        {"book_title": list(books_in_cluster), "similarity_score": similarities}
    )

    # Sorting all the books by similarity score and recommend them in the ascending order as they occur
    # there can be 1 as well for multiple books, only top 'n' are recommended
    recommendations = (
        similarities_df.sort_values("similarity_score", ascending=False)
        .head(10)["book_title"]
        .values.tolist()
    )
    recommendations_df = pd.DataFrame({"book_title": recommendations})
    recommendations_df = recommendations_df.reset_index(drop=True)
    books_list = recommendations_df["book_title"].values.tolist()
    # print(books_list)
    new_df = df[df["book_title"].isin(books_list)]
    return new_df


"""
a = recommend("Rules of Attraction")
print(a)
"""
