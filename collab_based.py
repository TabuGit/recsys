import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

user_movie = ratings.pivot_table(
    index='userId', columns='movieId', values='rating'
).fillna(0)

matrix = user_movie.values
user_means = np.mean(matrix, axis=1)
matrix_demeaned = matrix - user_means.reshape(-1, 1)

U, sigma, Vt = svds(matrix_demeaned, k=20)
sigma = np.diag(sigma)

predicted = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1)
preds_df = pd.DataFrame(predicted, columns=user_movie.columns, index=user_movie.index)

def recommend_for_user(user_id, n=5):
    if user_id not in preds_df.index:
        return []
    watched = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    user_preds = preds_df.loc[user_id].sort_values(ascending=False)
    user_preds = user_preds[~user_preds.index.isin(watched)].head(n)
    return movies[movies['movieId'].isin(user_preds.index)]['title']

for uid in [1, 5]:
    print(f"\nUser {uid}:")
    for title in recommend_for_user(uid, 5):
        print(f"  -> {title}")