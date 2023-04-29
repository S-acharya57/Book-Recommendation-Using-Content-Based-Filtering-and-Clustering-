import numpy as np
import pandas as pd
import pickle


def recommend(user_id, cluster_index):
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

    with open("kmeanstemp.pkl", "rb") as f:
        kmeans_temp = pickle.load(f)

    cluster_no = cluster_index
    cluster_books = df.iloc[np.where(kmeans_temp.labels_ == cluster_no)[0]][
        "book_title"
    ].tolist()
    temp = 0
    l = len(cluster_books)
    num_users = 2000
    num_items = l

    # set probability of zero
    p_zeros = 0.5

    # generate random matrix
    user_item_matrix = np.random.choice(
        [0, 1, 2, 3, 4, 5],
        size=(num_users, num_items),
        p=[p_zeros, 0.1, 0.1, 0.1, 0.1, 0.1],
    )
    # Create a DataFrame from the matrix

    cluster_no_int = cluster_no

    cluster_books = df.iloc[np.where(kmeans_temp.labels_ == cluster_no_int)[0]][
        "book_title"
    ].tolist()

    l = len(cluster_books)
    print(f"\nlength of books {l}")

    def matrix_factorization(R, P, Q, K, steps=5, alpha=0.002, beta=0.02):
        """
        R: rating matrix
        P: |U| * K (User features matrix)
        Q: |D| * K (Item features matrix)
        K: latent features
        steps: iterations
        alpha: learning rate
        beta: regularization parameter"""
        Q = Q.T

        for step in range(steps):
            print(f"First loop {step}remaining {steps -step}")
            for i in range(len(R)):
                # print(f"Second loop {i}remaining {len(R) -i}")
                for j in range(len(R[i])):
                    # print(f"Third loop {j}remaining {len(R[i]) -j}")
                    if R[i][j] > 0:
                        # calculate error
                        eij = R[i][j] - np.dot(P[i, :], Q[:, j])

                        for k in range(K):
                            # print(f"Fourth loop {k}")
                            # calculate gradient with a and beta parameter
                            P[i][k] = P[i][k] + alpha * (
                                2 * eij * Q[k][j] - beta * P[i][k]
                            )
                            Q[k][j] = Q[k][j] + alpha * (
                                2 * eij * P[i][k] - beta * Q[k][j]
                            )

            eR = np.dot(P, Q)
            e = 0
            for i in range(len(R)):
                # print(f"Error First loop {i}, remaining {len(R) -i}")
                for j in range(len(R[i])):
                    # print(f"Error Second loop {i}, remaining {len(R[i]) -j}")
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                        for k in range(K):
                            e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
                            # print(f"Error {e}")
            # 0.001: local minimum
            if e < 0.1:
                print(f"Breaking as e is {e}")
                break
        return P, Q.T

    R = user_item_matrix
    # N: num of User
    N = len(R)
    # M: num of Movie
    M = len(R[0])
    # Num of Features
    K = 3

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    nP, nQ = matrix_factorization(R, P, Q, K)

    nR = np.dot(nP, nQ.T)

    m = user_id
    print(m)
    print(nR[m, :])
    predictions = nR[m, :]
    print(predictions, predictions.shape)

    indices = np.where(predictions > 3)[0]
    indices_descending = indices[np.argsort(-predictions[indices])]

    cluster_books = df.iloc[np.where(kmeans_temp.labels_ == 12)[0]][
        "book_title"
    ].tolist()
    l = len(cluster_books)

    book_index_to_title = {index: title for index, title in enumerate(cluster_books)}

    top_books = [book_index_to_title[index] for index in indices_descending]
    new_df = df[df["book_title"].isin(top_books)]
    return new_df


#    for i, book_title in enumerate(top_books[:40], 1):
#        print(f"{i}. {book_title}")

# a = recommend(1984, 3)
# print(a)
