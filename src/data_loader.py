# funkcje do wczytywania danych z data/raw/ i zapisywania do data/processed/.

import pandas as pd

def read_csv_files():
    """
    Function reads csv files.
    Function returns 4 dataframes:
        - links,
        - movies,
        - ratings,
        - tags.
    """

    links = pd.read_csv("links.csv")
    movies =  pd.read_csv("movies.csv")
    ratings =  pd.read_csv("ratings.csv")
    tags =  pd.read_csv("tags.csv")

    return links, movies, ratings, tags

def cleaning_raw_data(links, movies, ratings, tags):
    """
    Function spots missing values in links and tags dataframes 
    and deletes data corresponding to those missing values in every dataframe.
    """
    links_nans = links[links.isna().any(axis=1)]
    tags_nans = tags[tags.isna().any(axis=1)]

    # we create list of movies IDs from links_nans and tags_nans
    movieId_from_links = links_nans['movieId'].values.tolist()
    movieId_from_tags = tags_nans['movieId'].values.tolist()

    # we get only unique values from list movieId_from_links and movieId_from_tags
    movieId_from_links = list(set(movieId_from_links))
    movieId_from_tags = list(set(movieId_from_tags))
    
    # we concatenate movieId_from_links and movieId_from_tags lists
    moviesId_to_remove = movieId_from_links + movieId_from_tags
    moviesId_to_remove = list(set(moviesId_to_remove))

    links.drop(links[links['movieId'].isin(moviesId_to_remove)].index, inplace = True)
    movies.drop(movies[movies['movieId'].isin(moviesId_to_remove)].index, inplace = True)
    ratings.drop(ratings[ratings['movieId'].isin(moviesId_to_remove)].index, inplace = True)
    tags.drop(tags[tags['movieId'].isin(moviesId_to_remove)].index, inplace = True)

    return links, movies, ratings, tags

def delete_inactive_users(ratings, inactive_users):
    """
    Function deletes users with less than given number of reviews.
    INPUT:
        - inactive_users - integer, number of reviews a user must have at least in order to not be deleted from database 
    """
    # inactive_users - variable indicating how many reviews at least a used must have not to be deleted
    inactive_users = inactive_users

    counts = ratings['userId'].value_counts()

    # we find users with less than indicated number of reviews
    rare_users = counts[counts < inactive_users].index

    # we print number of users with less than {inactive_users} reviews
    print(f"Users with less than {inactive_users} reviews: {list(rare_users)}")