import pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

stats = ratings.groupby('movieId').agg(
    mean_rating=('rating', 'mean'),
    count=('rating', 'count')
).reset_index()

popular = stats[stats['count'] >= 50]
popular = popular.sort_values('mean_rating', ascending=False).head(10)
popular = popular.merge(movies, on='movieId')

for _, row in popular.iterrows():
    print(f"{row['title']:55s} | {row['mean_rating']:.2f} | {row['count']}")