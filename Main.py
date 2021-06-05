import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Models import compute_normalize_map
import ast
from sklearn.ensemble import RandomForestRegressor
import pickle
from utils import add_dummies_to_end

TO_DROP = ["homepage",
                  "original_title",
                  "overview",
                  "production_countries",
                  "spoken_languages",
                  "status",
                  "tagline",
                  "title",
                  "belongs_to_collection",
                  "id",
                  ]

TO_AVERAGE = ["cast", "crew", "keywords", "genres", "production_companies"]
TO_NORMALIZE = ["budget", "vote_count", "runtime"]


def split_data(dataset):
    train_np = np.array(dataset.values, dtype=np.float32)
    X_train = train_np[:, :-1]
    y_train = train_np[:, -1]
    return X_train, y_train


def get_normalized_feature(dataset, name, normalize_map):
    for idx, line in enumerate(dataset[name]):
        if type(line) == str:
            l = ast.literal_eval(line)
            max_val = 0
            sum_val = 0
            for a in l:
                if a is not None:
                    if normalize_map.get(a["name"]) is not None:
                        element = normalize_map.get(a["name"])
                        sum_val += element
                        max_val = element if max_val < element else max_val
            if len(l) != 0:
                dataset.iloc[idx, dataset.columns.get_loc(name)] = (sum_val / len(l)) + 0.8 * max_val
            else:
                dataset.iloc[idx, dataset.columns.get_loc(name)] = 0
    return dataset


def add_productions():
    address = []
    max_val = 0
    for line in data["production_companies"]:
        if type(line) == str:
            l = ast.literal_eval(line)
            current = len(l)
            address.append(current)
            if max_val < current:
                max_val = current

    return list(map(lambda x: x / max_val, address))


def normalized_non_list_feature(x, max_val=None):
    x = data[x].values.reshape(-1, 1)
    if max_val is None:
        max_val = np.max(x)
    return pd.DataFrame(x / max_val), max_val


def normalizations(dataset):
    df_language = dataset["original_language"]
    dataset.loc[(df_language != 'en') & (df_language != 'fr') & (df_language != 'hi') & (df_language != 'ru') &
                (df_language != 'es') & (df_language != 'es') & (df_language != 'ja') & (df_language != 'it'),
                'original_language'] = 'other'

    dataset['year'] = pd.DatetimeIndex(dataset['release_date']).year
    dataset['Month'] = pd.DatetimeIndex(dataset['release_date']).month

    lang = ['en', 'es', 'fr', 'hi', 'it', 'ja', 'other', 'ru']
    dataset = add_dummies_to_end(dataset, 'original_language', lang)

    months = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7., 8., 9., 10., 11., 12.]
    dataset = add_dummies_to_end(dataset, 'Month', months)

    dataset.fillna(0, inplace=True)
    dataset = dataset.drop(columns=['original_language'])
    dataset = dataset.drop(columns=['release_date'])
    dataset = dataset.drop(columns=['Month'])

    return dataset


def drops(dataset):
    dataset.drop(TO_DROP, axis='columns', inplace=True)
    return dataset


def preprocess(dataset, train=True, normalization_values=None):
    dataset.fillna(0, inplace=True)
    dataset = drops(dataset)
    data["num_of_productions"] = add_productions()
    dataset = normalizations(dataset)
    if train:
        normalization_values = []
        for feature in TO_NORMALIZE:
            dataset[feature], max_val = normalized_non_list_feature(feature)
            normalization_values.append(max_val)
        return dataset, normalization_values
    else:
        for idx, feature in enumerate(TO_NORMALIZE):
            dataset[feature], _ = normalized_non_list_feature(feature, normalization_values[idx])
    return dataset


# training ---------------
data = pd.read_csv("movies_dataset.csv", na_values=['no info', '.']).copy()
# data2 = pd.read_csv("task1/movies_dataset_part2.csv", na_values=['no info', '.']).copy()
# data = pd.concat([data, data2], axis=0)
dataset, normalization_values = preprocess(data)

dataset_for_revenue = dataset[[col for col in dataset.columns if col != 'revenue'] + ['revenue']].copy()
dataset_for_revenue.drop("vote_average", axis='columns', inplace=True)
dataset_for_votes = dataset[[col for col in dataset.columns if col != 'vote_average'] + ['vote_average']].copy()
dataset_for_votes.drop("revenue", axis='columns', inplace=True)


def process_training(dataset, target):
    m = dict()
    for feature in TO_AVERAGE:
        m[feature] = compute_normalize_map(dataset, feature, target)
        dataset = get_normalized_feature(dataset, feature, m[feature])
    return dataset, m


dataset_for_revenue, m_revenue = process_training(dataset_for_revenue, "revenue")
dataset_for_votes, m_average = process_training(dataset_for_votes, "vote_average")


print("------- train random forest ---------")
# dataset.drop(["keywords", "runtime"])
X_train, y_train = split_data(dataset_for_revenue)
revenue_model = RandomForestRegressor(max_depth=8, random_state=4)
revenue_model.fit(X_train, y_train)


X_train, y_train = split_data(dataset_for_votes)
vote_average_model = RandomForestRegressor(max_depth=8, random_state=4)
vote_average_model.fit(X_train, y_train)


l = {"revenue": m_revenue, "vote_average": m_average}

with open('data.pkl', 'wb') as f:
    pickle.dump([normalization_values, l], f)

with open('revenue_model.pkl', 'wb') as f:
    pickle.dump(revenue_model, f)

with open('vote_model.pkl', 'wb') as f:
    pickle.dump(vote_average_model, f)
