# funkcje do przetwarzania danych i tworzenia cech.

#======================================================================================================================
#====================================================== IMPORT =======================================================#
#======================================================================================================================

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split

#======================================================================================================================
#=================================================== VISUALISATION ===================================================#
#======================================================================================================================
    
def plot_genre_distribution(df, column, title):
    """
    Plots a bar chart showing the distribution of movie genres
    and prints the genre counts below the chart.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    column : str
        Name of the column containing genre information.
        Genres can be single values or multiple values separated by '|'.
    title : str
        Title of the plot.

    Returns
    -------
    None
        This function displays a plot and prints genre counts,
        but does not return any value.
    """
    # separation of film genres
    genres_series = df[column].dropna().str.split('|').explode()

    # counting
    genre_counts = genres_series.value_counts().sort_values(ascending=False)

    # chart
    plt.figure(figsize=(12, 6))
    plt.bar(genre_counts.index, genre_counts.values)

    plt.title(title)
    plt.xlabel("genre")
    plt.ylabel("Amount")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

    # print result
    print("\nGenre counts:\n")
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")



def plot_value_counts(df, column, title, xlabel):
    """
    Plots a bar chart showing value counts for a given column
    and prints the counts below the chart.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    column : str
        Column name to analyze.
    title : str
        Title of the plot.
    xlabel : str
        Label for the x-axis.

    Returns
    -------
    None
        Displays a plot and prints value counts.
    """

    ratings_order = sorted(df[column].unique())

    counts = df[column].value_counts().reindex(ratings_order, fill_value=0)

    plt.figure(figsize=(12, 6))
    plt.bar(list(map(str, ratings_order)), counts.values)

    plt.xlabel(xlabel)
    plt.ylabel("Amount")
    plt.title(title)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # print result
    print(f"\nValue counts for '{column}':\n")
    for value, count in counts.items():
        print(f"{value}: {count}")

#======================================================================================================================
#================================================ FEATURES ENGINEERING ===============================================#
#======================================================================================================================

#=================================================== CONTENT BASED ====================================================

def merge_movies_with_tags(movies, tags):
    """
    Merges movies with tags grouped by movieId.

    Parameters
    ----------
    movies : pandas.DataFrame
        DataFrame with columns: movieId, title, genres
    tags : pandas.DataFrame
        DataFrame with columns: movieId, tag

    Returns
    -------
    pandas.DataFrame
        Movies enriched with a 'tags' column (aggregated per movieId)
    """

    # tag grouping for each movie
    tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()

    # merge with movies
    merged = movies.merge(tags_grouped, on="movieId", how="left")

    return merged



def build_content_matrix(df):
    """
    Builds a TF-IDF feature matrix for a content-based recommendation system.

    The function combines movie genres and tags into a single textual representation,
    then transforms this text into a numerical matrix using TF-IDF vectorization.
    This matrix can be used to compute similarity between movies.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - 'genres' : str
            Movie genres separated by '|'
        - 'tag' : str
            User-generated tags associated with movies (can contain NaN values)

    Returns
    -------
    matrix : scipy.sparse matrix
        TF-IDF feature matrix representing each movie as a vector of features.
    vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        Fitted TF-IDF vectorizer used to transform the text data.
    """

    # genres as text
    df["genres_clean"] = df["genres"].str.replace("|", " ")

    # no tags => empty string
    df["tag"] = df["tag"].fillna("")

    # we combine features
    df["content"] = df["genres_clean"] + " " + df["tag"]

    # TF-IDF
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df["content"])

    return matrix, vectorizer

#=============================== Collaborative Filtering with Matrix Factorization  ============================#

def prepare_cf_data(df_ratings, test_size=0.2, random_state=42):
    """
    Prepare dataset: split and map IDs.

    Parameters
    ---------------
    df_ratings : pd.DataFrame
    test_size : float
    random_state : int

    Returns
    --------
    train_df, test_df, user_map, movie_map
    """
    train_df, test_df = train_test_split(
        df_ratings, test_size=test_size, random_state=random_state
    )

    user_ids = train_df['userId'].unique()
    movie_ids = train_df['movieId'].unique()

    user_map = {id: i for i, id in enumerate(user_ids)}
    movie_map = {id: i for i, id in enumerate(movie_ids)}

    train_df['u'] = train_df['userId'].map(user_map)
    train_df['i'] = train_df['movieId'].map(movie_map)

    test_df = test_df[
        test_df['userId'].isin(user_map) &
        test_df['movieId'].isin(movie_map)
    ].copy()

    test_df['u'] = test_df['userId'].map(user_map)
    test_df['i'] = test_df['movieId'].map(movie_map)

    return train_df, test_df, user_map, movie_map


def cf_to_numpy(train_df, test_df):
    """
    Convert DataFrames to NumPy arrays.

    Parameters
    ---------------
    train_df, test_df : pd.DataFrame

    Returns
    --------
    train_users, train_items, train_ratings,
    test_users, test_items, test_ratings
    """
    return (
        train_df['u'].values,
        train_df['i'].values,
        train_df['rating'].values,
        test_df['u'].values,
        test_df['i'].values,
        test_df['rating'].values
    )