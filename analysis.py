# Imports
import pandas as pd
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle
from os.path import exists

# helper functions

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


# define class to complete analysis
class FreshModel(object):
    def __init__(self,
                 movies_fname: str = 'movie_info.tsv',
                 reviews_fname: str = 'reviews.tsv',
                 random_state=23):
        self.movies_fname = movies_fname
        self.reviews_fname = reviews_fname
        self.random_state = random_state
        self.data = None
        self.unique_genres = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_fname = 'X_train.pkl'
        self.X_test_fname = 'X_test.pkl'
        self.y_train_fname = 'y_train.pkl'
        self.y_test_fname = 'y_test.pkl'
        self.svc_accuracy = None
        self.rf_accuracy = None
        self.y_pred_svm = None
        self.y_pred_rf = None
        self.X_test_genres = None
        self.feature_names = None
        self.svc = None
        self.rf = None

        # featurization pipelines
        text_pipe_rev = Pipeline(steps=[
            # BOW matrix
            ('vectorize', TfidfVectorizer(stop_words='english',
                                          ngram_range=(1, 2),
                                          min_df=5,
                                          max_df=0.85,
                                          max_features=20000)),

            # topic modeling
            ('topic_model', TruncatedSVD(n_components=50, random_state=self.random_state))
        ])

        text_pipe_syn = Pipeline(steps=[
            # BOW matrix
            ('vectorize', TfidfVectorizer(stop_words='english',
                                          ngram_range=(1, 2),
                                          min_df=5,
                                          max_df=0.85,
                                          max_features=20000)),

            # topic modeling
            ('topic_model', TruncatedSVD(n_components=50, random_state=self.random_state))
        ])

        self.ct = ColumnTransformer(
            transformers=[
                ('text_review', text_pipe_rev, 'review'),
                ('text_synopsis', text_pipe_syn, 'synopsis')
            ],
            remainder='passthrough',
            n_jobs=-1
        )

    def load_data_initial(self):
        movies = pd.read_csv('movie_info.tsv', sep='\t')
        print(f'Movie file columns: {movies.columns}')  # ['id', 'synopsis', 'rating', 'genre', 'director', 'writer',
        # 'theater_date', 'dvd_date', 'currency', 'box_office', 'runtime', 'studio']

        reviews = pd.read_csv('reviews.tsv', sep='\t', encoding='ISO-8859-1', engine='python')
        print(f'Review file columns: {reviews.columns}')  # ['id', 'review', 'rating', 'fresh', 'critic',
        # 'top_critic', 'publisher', 'date']
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
        print('Number of fresh & rotten films:')
        print(data_selected['fresh'].value_counts())
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
        exploded_series = data_selected.genre.explode()
        self.unique_genres = exploded_series.unique()
        with open('unique_genres.pkl', 'wb') as f:
            pickle.dump(self.unique_genres, f)
        genres_df = pd.get_dummies(exploded_series)
        genres_df = genres_df.reset_index()
        genres_df = genres_df.groupby('index').sum()  # collapse rows
        data_selected = data_selected.join(genres_df)  # rejoin with original df
        data_selected.drop(columns=['genre'], inplace=True)

        # lemmatize text
        data_selected['review'] = data_selected['review'].apply(lambda x: get_lemmas(x))
        data_selected['synopsis'] = data_selected['synopsis'].apply(lambda x: get_lemmas(x))

        self.data = data_selected.copy()
        with open('data.pkl', 'wb') as f:
            pickle.dump(self.data, f)

        # get train/test sets
        X = data_selected.drop(columns=['fresh'])  # pd.DataFrame
        y = data_selected['fresh']  # pd.Series

        self.feature_names = X.columns.to_list()
        feature_names = []
        for el in self.feature_names[:2]:
            feature_names = feature_names + [f'{el}_topic{i+1}' for i in range(50)]
        self.feature_names = feature_names + self.feature_names[2:]
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.random_state)
        print(f'The size of the training set is {self.X_train.shape[0]}.')

        # save X_test genre information
        self.X_test_genres = self.X_test[self.unique_genres]
        with open('X_test_genres.pkl', 'wb') as f:
            pickle.dump(self.X_test_genres, f)

        # fit & transform X_train dataframe into feature matrix for model training
        self.X_train = self.ct.fit_transform(self.X_train)
        print('Fit & transformed X_train.')

        # transform X_test dataframe into feature matrix for model testing
        self.X_test = self.ct.transform(self.X_test)
        print('Transformed X_test.')

        # save data files to local directory
        with open(self.X_train_fname, 'wb') as f:
            pickle.dump(self.X_train, f)

        with open(self.X_test_fname, 'wb') as f:
            pickle.dump(self.X_test, f)

        with open(self.y_train_fname, 'wb') as f:
            pickle.dump(self.y_train, f)

        with open(self.y_test_fname, 'wb') as f:
            pickle.dump(self.y_test, f)

    def load_data_intermediate(self):
        with open(self.X_train_fname, 'rb') as f:
            self.X_train = pickle.load(f)
        print(f'Shape of X_train: {self.X_train.shape}')

        with open(self.X_test_fname, 'rb') as f:
            self.X_test = pickle.load(f)

        with open(self.y_train_fname, 'rb') as f:
            self.y_train = pickle.load(f)

        with open(self.y_test_fname, 'rb') as f:
            self.y_test = pickle.load(f)

        with open('data.pkl', 'rb') as f:
            self.data = pickle.load(f)

        with open('unique_genres.pkl', 'rb') as f:
            self.unique_genres = pickle.load(f)
        print(f'Number of genres: {len(self.unique_genres)}')

        with open('X_test_genres.pkl', 'rb') as f:
            self.X_test_genres = pickle.load(f)

        with open('feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)

    def model_train_test(self):
        # linear model
        # svc = LinearSVC(penalty='l1', random_state=self.random_state)
        svc = SVC(random_state=self.random_state)
        # svc_param_grid = {
        #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #     'C': [0.1, 1, 10]
        # }
        # svc_estimator = GridSearchCV(svc, svc_param_grid, n_jobs=-1)
        # svc_estimator.fit(self.X_train, self.y_train)
        # y_pred_svm = svc_estimator.predict(self.X_test)
        svc.fit(self.X_train, self.y_train)
        self.svc = svc
        print('Fit SVC model.')
        with open('svc.pkl', 'wb') as f:
            pickle.dump(svc, f)
        self.y_pred_svm = svc.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred_svm, zero_division=0))
        self.svc_accuracy = accuracy_score(self.y_test, self.y_pred_svm)

        # ensemble decision tree model
        rf = RandomForestClassifier(random_state=self.random_state)
        # rf_param_grid = {
        #     'n_estimators': [50, 100, 200]
        # }
        # rf_estimator = GridSearchCV(rf, rf_param_grid, n_jobs=-1)
        # rf_estimator.fit(X_train, y_train)
        # y_pred_rf = rf_estimator.predict(X_test)
        rf.fit(self.X_train, self.y_train)
        self.rf = rf
        with open('rf.pkl', 'wb') as f:
            pickle.dump(rf, f)
        print('Fit Random Forest model.')
        self.y_pred_rf = rf.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred_rf, zero_division=0))
        self.rf_accuracy = accuracy_score(self.y_test, self.y_pred_rf)

    def model_test(self):
        with open('svc.pkl', 'rb') as f:
            svc = pickle.load(f)
        self.svc = svc
        self.y_pred_svm = svc.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred_svm, zero_division=0))
        self.svc_accuracy = accuracy_score(self.y_test, self.y_pred_svm)

        with open('rf.pkl', 'rb') as f:
            rf = pickle.load(f)
        self.rf = rf
        self.y_pred_rf = rf.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred_rf, zero_division=0))
        self.rf_accuracy = accuracy_score(self.y_test, self.y_pred_rf)

    def results_by_genre(self):
        # Task 2: Interrogate the predictor
        results_df = pd.concat([self.X_test_genres, self.y_test], axis=1)
        if self.svc_accuracy < self.rf_accuracy:
            results_df['y_pred'] = self.y_pred_rf
        elif self.svc_accuracy >= self.rf_accuracy:
            results_df['y_pred'] = self.y_pred_svm

        # genres:
        genre_accuracy = []
        genre_f1 = []
        genre_n = []
        for g in self.unique_genres:
            filtered_results = results_df[results_df[g] == 1]
            n_obs = len(filtered_results)
            acc = accuracy_score(filtered_results['fresh'], filtered_results['y_pred'])
            f1 = f1_score(filtered_results['fresh'], filtered_results['y_pred'])
            genre_n.append(n_obs)
            genre_accuracy.append(acc)
            genre_f1.append(f1)

        genre_results = pd.DataFrame({
            'genre': self.unique_genres,
            'count': genre_n,
            'accuracy': genre_accuracy,
            'f1': genre_f1
        })

        return genre_results

    def get_feature_importance(self):
        rf_importances = self.rf.feature_importances_
        importance_df = pd.DataFrame({'var': self.feature_names, 'importance': rf_importances})
        importance_df.sort_values(by='importance', ascending=False, inplace=True, ignore_index=True)

        return importance_df

    def main(self):
        if all([exists(self.X_train_fname),
                exists(self.X_test_fname),
                exists(self.y_train_fname),
                exists(self.y_test_fname)]):
            self.load_data_intermediate()
        else:
            self.load_data_initial()

        if all([exists('svc.pkl'), exists('rf.pkl')]):
            self.model_test()
        else:
            self.model_train_test()

        genre_results_df = self.results_by_genre()
        print(genre_results_df)

        if self.svc_accuracy < self.rf_accuracy:
            print(self.get_feature_importance())


if __name__ == '__main__':
    analysis = FreshModel()
    analysis.main()
    if analysis.rf_accuracy > analysis.svc_accuracy:
        feature_importance_df = analysis.get_feature_importance()
        feature_importance_df.to_csv('feature_importance.csv')

        genre_results_df = analysis.results_by_genre()
        genre_results_df.to_csv('genre_results.csv')

