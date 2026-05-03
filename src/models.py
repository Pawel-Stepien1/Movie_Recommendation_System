# funkcje/klasy do budowy, trenowania i ewaluacji modeli ML.

#======================================================================================================================
#====================================================== IMPORT =======================================================#
#======================================================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


# ======= Models =======

#======================================================================================================================
#=============================== Collaborative Filtering with Matrix Factorization (SGD)  ============================#
#======================================================================================================================

class MF_SGD:
    """
    Matrix Factorization using Stochastic Gradient Descent.

    Input
    ---------------
    k : int
        Number of latent factors
    lr : float
        Learning rate
    reg : float
        Regularization strength
    epochs : int
        Number of training epochs

    Output
    --------
    Trained MF model
    """

    def __init__(self, k=50, lr=0.01, reg=0.02, epochs=10, random_state=42):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, train_users, train_items, train_ratings, n_users, n_items):
        """
        Train MF model.

        Input
        ---------------
        train_users : np.ndarray
        train_items : np.ndarray
        train_ratings : np.ndarray
        n_users : int
        n_items : int

        Output
        --------
        self
        """
        np.random.seed(self.random_state)

        self.P = np.random.normal(0, 0.1, (n_users, self.k)).astype(np.float32)
        self.Q = np.random.normal(0, 0.1, (n_items, self.k)).astype(np.float32)

        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)

        self.mu = train_ratings.mean()

        for epoch in range(self.epochs):
            indices = np.arange(len(train_users))
            np.random.shuffle(indices)

            train_users = train_users[indices]
            train_items = train_items[indices]
            train_ratings = train_ratings[indices]

            total_error = 0

            for idx in range(len(train_users)):
                u = train_users[idx]
                i = train_items[idx]
                r = train_ratings[idx]

                pred = self.predict_single(u, i)
                err = r - pred

                # bias update
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # latent update
                Pu = self.P[u]
                Qi = self.Q[i]

                self.P[u] += self.lr * (err * Qi - self.reg * Pu)
                self.Q[i] += self.lr * (err * Pu - self.reg * Qi)

                total_error += err**2

            rmse = np.sqrt(total_error / len(train_users))
            print(f"[MF-SGD] Epoch {epoch+1}: RMSE = {rmse:.4f}")

        return self

    def predict_single(self, u, i):
        """
        Predict rating for a single user-item pair.

        Input
        ---------------
        u : int
        i : int

        Output
        --------
        float
        """
        return self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])

    def predict_batch(self, users, items):
        """
        Predict ratings for multiple user-item pairs.

        Input
        ---------------
        users : np.ndarray
        items : np.ndarray

        Output
        --------
        np.ndarray
        """
        return (
            self.mu
            + self.bu[users]
            + self.bi[items]
            + np.sum(self.P[users] * self.Q[items], axis=1)
        )

    def recommend(self, user_id, user_map, movie_map, df_movies, top_n=10):
        """
        Generate Top-N recommendations for a user.

        Input
        ---------------
        user_id : int
        user_map : dict
        movie_map : dict
        df_movies : pd.DataFrame
        top_n : int

        Output
        --------
        pd.DataFrame
            Recommended movies (titles)
        """
        if user_id not in user_map:
            return pd.DataFrame()

        u = user_map[user_id]

        scores = self.mu + self.bu[u] + self.bi + self.P[u].dot(self.Q.T)

        top_items = np.argsort(scores)[-top_n:][::-1]

        inv_movie_map = {v: k for k, v in movie_map.items()}
        movie_ids = [inv_movie_map[i] for i in top_items]

        return df_movies[df_movies['movieId'].isin(movie_ids)][['title']]


# =========================
# DATA PREPARATION HELPERS
# =========================

def prepare_cf_data(df_ratings, test_size=0.2, random_state=42):
    """
    Prepare dataset: split and map IDs.

    Input
    ---------------
    df_ratings : pd.DataFrame
    test_size : float
    random_state : int

    Output
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

    Input
    ---------------
    train_df, test_df : pd.DataFrame

    Output
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

# == SVD ==
class MF_SVD:
    """
    Model-based Collaborative Filtering using Singular Value Decomposition (SVD).
    This implementation applies user-mean normalization to handle the cold-start 
    problem for existing users and optimizes performance for large datasets.

    Input
    ----------
    k : int
        Number of latent factors (rank of the approximation).
    """

    def __init__(self, k=50):
        self.k = k
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_means = None
        self.global_mean = None

    def fit(self, train_df, n_users, n_movies):
        """
        Train the SVD model by decomposing the normalized user-item matrix.
        Matches the 'u' and 'i' column naming convention from prepare_cf_data.

        Input
        ----------
        train_df : pd.DataFrame
            DataFrame containing columns ['u', 'i', 'rating'].
        n_users : int
            Total number of unique users.
        n_movies : int
            Total number of unique movies.

        Output
        -----------
        self : MF_SVD
            The trained model instance.
        """
        # Set column names used by prepare_cf_data
        u_col, i_col = 'u', 'i'

        # Calculate global and user-specific means for normalization
        self.global_mean = train_df['rating'].mean()
        self.user_means = train_df.groupby(u_col)['rating'].mean()

        # Normalize ratings: r_norm = r - user_mean
        # Fill missing user means with global mean to ensure robustness
        train_df_norm = train_df.copy()
        train_df_norm['rating_norm'] = train_df_norm['rating'] - \
            train_df_norm[u_col].map(self.user_means).fillna(self.global_mean)

        # Create sparse matrix for memory-efficient decomposition
        train_sparse = csr_matrix(
            (train_df_norm['rating_norm'], (train_df_norm[u_col], train_df_norm[i_col])),
            shape=(n_users, n_movies)
        )

        # Perform Spectral Decomposition (SVD)
        # svds returns U, s, and Vt matrices
        u, s, vt = svds(train_sparse.astype(float), k=self.k)
        
        # Sort latent factors in descending order by singular values
        idx = np.argsort(s)[::-1]
        self.U = u[:, idx]
        self.sigma = np.diag(s[idx])
        self.Vt = vt[idx, :]

        return self

    def predict_batch(self, user_indices, movie_indices):
        """
        Predict ratings for a batch of user-item pairs using matrix dot products.

        Input
        ----------
        user_indices : np.ndarray
            Array of mapped user indices ('u').
        movie_indices : np.ndarray
            Array of mapped movie indices ('i').

        Output
        -----------
        predictions : np.ndarray
            Predicted ratings clipped to the standard [0.5, 5.0] scale.
        """
        # Retrieve baseline means for each user in the batch
        u_means = np.array([self.user_means.get(u, self.global_mean) for u in user_indices])
        
        # Compute SVD component: U[u] * Sigma * Vt[:, m]
        # Using list comprehension for per-pair dot product calculation
        svd_preds = np.array([
            np.dot(np.dot(self.U[u, :], self.sigma), self.Vt[:, m])
            for u, m in zip(user_indices, movie_indices)
        ])
        
        # Final prediction = baseline mean + SVD deviation
        return np.clip(u_means + svd_preds, 0.5, 5.0)

    def recommend(self, user_id, user_map, movie_map, df_movies, top_n=10):
        """
        Generate Top-N recommendations for a specific user ID.

        Input
        ----------
        user_id : int
            Original user ID from the source dataset.
        user_map : dict
            Mapping from original user IDs to indices.
        movie_map : dict
            Mapping from original movie IDs to indices.
        df_movies : pd.DataFrame
            DataFrame containing movie titles and metadata.
        top_n : int
            Number of top-rated movies to return.

        Output
        -----------
        recommendations : pd.DataFrame
            DataFrame containing top-N recommended movie titles.
        """
        if user_id not in user_map:
            return pd.DataFrame()

        u_idx = user_map[user_id]
        u_mean = self.user_means.get(u_idx, self.global_mean)
        
        # Predict ratings for all items simultaneously using matrix multiplication
        # Preds = user_mean + (U_vector * Sigma) * Vt_matrix
        all_svd_diffs = np.dot(np.dot(self.U[u_idx, :], self.sigma), self.Vt)
        all_preds = u_mean + all_svd_diffs
        
        # Sort indices by predicted scores in descending order
        top_indices = np.argsort(all_preds)[::-1][:top_n]
        
        # Reverse map indices back to original movie IDs
        inv_movie_map = {v: k for k, v in movie_map.items()}
        movie_ids = [inv_movie_map[idx] for idx in top_indices]
        
        # Return movie titles from metadata
        return df_movies[df_movies['movieId'].isin(movie_ids)][['title']]
    
#======================================================================================================================
#=================================================== CONTENT BASED ====================================================
#======================================================================================================================

def content_based_recommendations(movie_id, df, matrix, top_n=10):
    """
    Recommends movies similar to a given movie using a content-based approach
    based on cosine similarity of TF-IDF feature vectors.

    The function finds the target movie in the dataset, computes cosine similarity
    between its feature vector and all other movies, and returns the most similar ones.

    Parameters
    ----------
    movie_id : int
        ID of the reference movie for which recommendations are generated.
    df : pandas.DataFrame
        DataFrame containing movie metadata. Must include:
        - 'movieId' : int
        - 'title' : str
    matrix : scipy.sparse matrix
        TF-IDF feature matrix where each row corresponds to a movie.
    top_n : int, optional (default=10)
        Number of recommended movies to return.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the top-N most similar movies with columns:
        - 'movieId'
        - 'title'
    """

    idx = df.index[df["movieId"] == movie_id][0]

    similarity = cosine_similarity(matrix[idx], matrix).flatten()

    similar_indices = similarity.argsort()[::-1][1:top_n+1]

    return df.iloc[similar_indices][["movieId", "title"]]


def user_content_recommendations(user_id, ratings, movies, matrix, top_n=10, min_rating=3.5):
    """
    Generates movie recommendations for a given user based on their entire viewing history
    using a content-based filtering approach.

    The function builds a user profile by aggregating TF-IDF feature vectors of movies
    that the user has positively rated (above a specified threshold). Each movie vector
    is weighted by the user's rating, creating a representation of the user's preferences.
    Then, cosine similarity is computed between the user profile and all movies to find
    the most relevant recommendations.

    Parameters
    ----------
    user_id : int
        ID of the user for whom recommendations are generated.
    ratings : pandas.DataFrame
        DataFrame containing user ratings with columns:
        - 'userId' : int
        - 'movieId' : int
        - 'rating' : float
    movies : pandas.DataFrame
        DataFrame containing movie metadata with columns:
        - 'movieId' : int
        - 'title' : str
    matrix : scipy.sparse matrix
        TF-IDF feature matrix where each row corresponds to a movie
        (aligned with the `movies` DataFrame).
    top_n : int, optional (default=10)
        Number of recommendations to return.
    min_rating : float, optional (default=3.5)
        Minimum rating threshold used to determine which movies are considered
        "liked" by the user.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the top-N recommended movies with columns:
        - 'movieId'
        - 'title'

    Notes
    -----
    - Movies already rated by the user are excluded from recommendations.
    - The function operates on sparse matrices for efficiency and avoids
      converting them to dense format to reduce memory usage.
    - If the user has no ratings above `min_rating`, an empty DataFrame is returned.
    """

    user_ratings = ratings[ratings["userId"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= min_rating]

    if liked.empty:
        return pd.DataFrame(columns=["movieId", "title"])

    # movie indexes
    movie_indices = movies.index[movies["movieId"].isin(liked["movieId"])]

    # weights (ratings)
    weights = liked.set_index("movieId").loc[
        movies.loc[movie_indices, "movieId"]
    ]["rating"].values

    user_matrix = matrix[movie_indices]

    user_profile = user_matrix.multiply(weights[:, np.newaxis]).sum(axis=0)

    user_profile = np.asarray(user_profile)

    # normalization
    user_profile = user_profile / np.sum(weights)

    # similarity
    similarity = cosine_similarity(user_profile, matrix).flatten()

    # filtering watched movies
    seen_movies = set(user_ratings["movieId"])

    indices = [
        i for i in similarity.argsort()[::-1]
        if movies.iloc[i]["movieId"] not in seen_movies
    ][:top_n]

    return movies.iloc[indices][["movieId", "title"]]