# funkcje do przetwarzania danych i tworzenia cech.

#======================================================================================================================
#====================================================== IMPORT =======================================================#
#======================================================================================================================

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

#=============================== Collaborative Filtering with Matrix Factorization (SGD)  ============================#