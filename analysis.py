# Imports
import pandas as pd
import re

# Load data
movies = pd.read_csv('movie_info.tsv', sep='\t')
movies.head()
print(movies.columns) # ['id', 'synopsis', 'rating', 'genre', 'director', 'writer', 'theater_date', 'dvd_date',
# 'currency', 'box_office', 'runtime', 'studio']

reviews = pd.read_csv('reviews.tsv', sep='\t', encoding='ISO-8859-1', engine='python')
reviews.head()
print(reviews.columns) # ['id', 'review', 'rating', 'fresh', 'critic', 'top_critic', 'publisher', 'date']
reviews.rename(columns={'rating': 'review_rating'}, inplace=True)

# join tables
data = reviews.merge(movies, on='id') # inner join

# select features: review, top_critic, synopsis, genre, box_office, runtime
# target: fresh
data_selected = data[[
    'fresh', 'review', 'synopsis', 'top_critic', 'genre', 'box_office', 'runtime'
]]

# drop rows where features or target are Null
data_selected = data_selected.dropna()
data_selected.reset_index(drop=True, inplace=True)

# clean data

# change box office to numeric
data_selected['box_office'] = data_selected['box_office'].apply(lambda x: int(re.sub('[^0-9]+', '', x)))

# change runtime to numeric
data_selected['runtime'] = data_selected['runtime'].apply(lambda x: int(re.sub('[^0-9]+', '', x)))

# change fresh to binary outcome
data_selected['fresh'] = data_selected['fresh'].apply(lambda x: int(x == 'fresh'))

# extract genres
data_selected['genre'] = data_selected['genre'].apply(lambda x: x.split('|'))
genres_df = pd.get_dummies(data_selected.genre.explode())
genres_df = genres_df.reset_index()
genres_df = genres_df.groupby('index').sum() # collapse rows
data_selected = data_selected.join(genres_df) # rejoin with original df
data_selected.drop(['genre'], inplace=True)

# BOW

