from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import json
import tempfile
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import urllib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# Read the raw json file
# with open('data_interview.json', 'r') as fp:
    # json_content = json.load(fp)

# Data Frame to store data
# df = pd.DataFrame()

# Set values in dataframe from json file
# for col in list(json_content.keys()):
    # df[col] = json_content[col]

# Write to CSV
# df.to_csv('raw_data.csv')

LABEL_COLUMN = "label"

# List of categorical columns
CATEGORICAL_COLUMNS = [
    'city',
    'phone',
    'business_service',
]

# List of continuous columns
CONTINUOUS_COLUMNS = [
    'service_rating',
    'extra_cost_pct',
    'customer_rating',
    'avg_extra_cost',
    'weekday_pct',
    'avg_use_of_service',
    'use_of_service_in_first_30_days',
    'recency'
]


def build_estimator(model_dir, model_type='wide'):
    tf.logging.set_verbosity(tf.logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    """Build an estimator."""

    # City Column
    city = tf.contrib.layers.sparse_column_with_keys(
        column_name="city", keys=['A', 'K', 'W'])

    # phone
    phone_values = ['iPhone', 'Android']
    phone = tf.contrib.layers.sparse_column_with_keys(
        column_name="phone", keys=phone_values)

    business_service = tf.contrib.layers.sparse_column_with_keys(
        column_name="business_service", keys=['0', '1'])


    # CONTINUOUS_COLUMNS
    service_rating = tf.contrib.layers.real_valued_column("service_rating")
    extra_cost_pct = tf.contrib.layers.real_valued_column("extra_cost_pct")
    customer_rating = tf.contrib.layers.real_valued_column("customer_rating")
    avg_extra_cost = tf.contrib.layers.real_valued_column("avg_extra_cost")
    weekday_pct = tf.contrib.layers.real_valued_column("weekday_pct")
    avg_use_of_service = tf.contrib.layers.real_valued_column("avg_use_of_service")
    use_of_service_in_first_30_days = tf.contrib.layers.real_valued_column("use_of_service_in_first_30_days")
    recency = tf.contrib.layers.real_valued_column("recency")

    deep_columns = [
        tf.contrib.layers.embedding_column(city, dimension=8),
        tf.contrib.layers.embedding_column(phone, dimension=8),
        tf.contrib.layers.embedding_column(business_service, dimension=8),
        service_rating,
        extra_cost_pct,
        customer_rating,
        avg_extra_cost,
        weekday_pct,
        avg_use_of_service,
        use_of_service_in_first_30_days,
        recency        
    ]
    
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       optimizer=opt,
                                       feature_columns=deep_columns,
                                       hidden_units=[512, 256, 128])
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    try:
        label = tf.constant(df[LABEL_COLUMN].values)
    except:
        label = None
    # Returns the feature columns and the label.
    return feature_cols, label


def input_fn_predict(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    # label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols


def train_and_eval(df, model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    # train_file_name, test_file_name = maybe_download(train_data, test_data)
    # dtype = {
    #     'bucked_income': str,
    #     'bucketed_age': str,
    #     'bucketed_terms': str
    # }
    # df = pd.read_csv('CSVs/training/disposition_model_input.csv')
    df_new = df
    # df_pos = df[df['is_retained'] == 1]
    # df_neg = df[df['is_retained'] == 0]
    # df_new = df_pos.sample(round(len(df_pos) / 2))
    # df_new = df_new.append(df_pos.sample(round(len(df_pos) / 2)))
    # df_new = df_new.append(df_pos.sample(round(len(df_pos) / 2)))
    # df_new = df_new.append(df_pos.sample(round(len(df_pos) / 2)))
    # df_new = df_new.append(df_neg.sample(len(df_pos)))
    # df_new = df_new.append(df_neg.sample(len(df_pos)))
    # df_new = df_new.append(df_neg.sample(len(df_pos)))
    # df_new = df_new.append(df_neg.sample(len(df_pos)))
    # df_new = df_new.append(df_neg.sample(len(df_pos)))

    for col in list(df.columns):
        print(col)
        print(df[col].head())
        print('~~~~~~~~~~~~~~~~~~~')

    varification_df = df.sample(1000)
    varification_df.to_csv('validation.csv', index=False)

    print(len(varification_df))
    print(len(df))

    df = df[~df.user_id.isin(list(varification_df.user_id.unique()))] 

    print(len(df))

    df = df.loc[:, df.columns != 'user_id']

    print(df.head())

    df_train, df_test, _, _ = train_test_split(
        df_new, df_new['is_retained'], train_size=0.7)

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    df_train[LABEL_COLUMN] = (
        df_train["is_retained"].apply(lambda x: x == 1)).astype(int)
    df_test[LABEL_COLUMN] = (
        df_test["is_retained"].apply(lambda x: x == 1)).astype(int)

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    return m


FLAGS = None


def main(_):

    df = pd.read_csv('raw_data.csv')

    start = datetime.strptime('1970-01-01', '%Y-%m-%d')

    # Get actual date from difference
    # df['signup_date'] = df['signup_date'].apply(lambda x: start + timedelta(days=x))
    df['last_service_use_date'] = df['last_service_use_date'].apply(lambda x: start + timedelta(days=x))

    df.rename(columns={'Unnamed: 0': 'user_id'}, inplace=True)

    # Get user's recency
    df['recency'] = df['last_service_use_date'].apply(lambda x: (df.last_service_use_date.max() - x).days)

    # Convert True False to 0 & 1
    df.loc[df['business_service'] == True, 'business_service'] = '1'
    df.loc[df['business_service'] == False, 'business_service'] = '0'

    df['is_retained'] = 0
    df.loc[df['last_service_use_date'].dt.month.isin([6,7]), 'is_retained'] = 1

    m = train_and_eval(df, FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                       FLAGS.train_data, FLAGS.test_data)
    # m = build_estimator(FLAGS.model_dir)
    # df = pd.read_csv('overall_ystory_june.csv')
    # predicted_values = list(m.predict(input_fn=lambda: input_fn(predict_df)))
    # probs = list(m.predict_proba(input_fn=lambda: input_fn(predict_df)))
    # print(predicted_values)
    # print(probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# we have used 10000 train steps article
