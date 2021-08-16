# Imports
import pandas as pd
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
movies = pd.read_csv('movie_info.tsv', sep='\t')
movies.head()
print(movies.columns)  # ['id', 'synopsis', 'rating', 'genre', 'director', 'writer', 'theater_date', 'dvd_date',
# 'currency', 'box_office', 'runtime', 'studio']

reviews = pd.read_csv('reviews.tsv', sep='\t', encoding='ISO-8859-1', engine='python')
reviews.head()
print(reviews.columns)  # ['id', 'review', 'rating', 'fresh', 'critic', 'top_critic', 'publisher', 'date']
reviews.rename(columns={'rating': 'review_rating'}, inplace=True)

# join tables
data = reviews.merge(movies, on='id')  # inner join

# select features: review, top_critic, synopsis, genre, box_office, runtime
# target: fresh
data_selected = data[[
    'fresh', 'review', 'synopsis', 'top_critic', 'genre', 'box_office', 'runtime'
]]

# drop rows where features or target are Null
data_selected = data_selected.dropna()

# drop any rows where text cols are Null
data_selected = data_selected[(data_selected.review != '') & (data_selected.synopsis != '')]

# reset index
data_selected.reset_index(drop=True, inplace=True)

# count num unique vals
data_selected['fresh'].value_counts()
# 1    18806
# 0    13169
# Name: fresh, dtype: int64


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
genres_df = genres_df.groupby('index').sum()  # collapse rows
data_selected = data_selected.join(genres_df)  # rejoin with original df
data_selected.drop(columns=['genre'], inplace=True)

# text lemmatization
lemmatizer = WordNetLemmatizer()


# to convert nltk_pos tags to wordnet-compatible PoS tags
def convert_pos_wordnet(tag):
    tag_abbr = tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }

    if tag_abbr in tag_dict:
        return tag_dict[tag_abbr]


def lemmatize_w_pos(word):
    tag = pos_tag([word])[0][1]
    pos_bool = tag[0].upper() in 'JNVR'

    if pos_bool:
        lemma = lemmatizer.lemmatize(word, convert_pos_wordnet(tag))
    else:
        lemma = lemmatizer.lemmatize(word)

    return lemma


def get_lemmas(string):
    tokens = word_tokenize(string)
    lemmas = [lemmatize_w_pos(tok) for tok in tokens]

    return ' '.join(lemmas)


data_selected['review'] = data_selected['review'].apply(lambda x: get_lemmas(x))
data_selected['synopsis'] = data_selected['synopsis'].apply(lambda x: get_lemmas(x))


# featurization pipelines
text_pipe = Pipeline(steps=[
    # BOW matrix
    ('vectorize', TfidfVectorizer(stop_words='english',
                                  ngram_range=(1, 2),
                                  min_df=5,
                                  max_df=0.85,
                                  max_features=20000)),

    # topic modeling
    ('topic_model', TruncatedSVD())
])

col_transforms = ColumnTransformer(
    transformers=[('text', text_pipe, ['review', 'synopsis'])],
    remainder='passthrough'
)

# get train/test sets
X = data_selected.drop(columns=['fresh'])  # pd.DataFrame
y = data_selected['fresh']  # pd.Series

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)
print(f'The size of the training set is {X_train.shape[0]}.')

# fit & transform X_train dataframe into feature matrix for model training
X_train = col_transforms.fit_transform(X_train)

# transform X_test dataframe into feature matrix for model testing
X_test = col_transforms.transform(X_test)

# modeling

# linear model
model = SVC(random_state=23)
svc_param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10]
}
svc_estimator = GridSearchCV(model, svc_param_grid)
svc_estimator.fit(X_train, y_train)
y_pred = svc_estimator.predict(X_test)
print(classification_report(y_test, y_pred))

# ensemble decision tree model
model = RandomForestClassifier(random_state=23)
rf_param_grid = {
    'n_estimators': [50, 100, 200]
}
rf_estimator = GridSearchCV(model, rf_param_grid)
rf_estimator.fit(X_train, y_train)
y_pred = rf_estimator.predict(X_test)
print(classification_report(y_test, y_pred))
