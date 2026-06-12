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
from sklearn.preprocessing import normalize, MinMaxScaler
from tqdm.notebook import tqdm


# ======= Models =======

#======================================================================================================================
#=============================== Collaborative Filtering with Matrix Factorization (SGD)  ============================#
#======================================================================================================================

class MF_SGD:
    """
    Matrix Factorization using Stochastic Gradient Descent.

    Parameters
    ---------------
    k : int
        Number of latent factors
    lr : float
        Learning rate
    reg : float
        Regularization strength
    epochs : int
        Number of training epochs

    Returns
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

        Parameters
        ---------------
        train_users : np.ndarray
        train_items : np.ndarray
        train_ratings : np.ndarray
        n_users : int
        n_items : int

        Returns
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
            print(f"[cf-sgd] Epoch {epoch+1}: RMSE = {rmse:.4f}")

        return self

    def predict_single(self, u, i):
        """
        Predict rating for a single user-item pair.

        Parameters
        ---------------
        u : int
        i : int

        Returns
        --------
        float
        """
        return self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])

    def predict(self, users, items):
        """
        Predict ratings for multiple user-item pairs.

        Parameters
        ---------------
        users : np.ndarray
        items : np.ndarray

        Returns
        --------
        np.ndarray
        """
        return (self.mu + self.bu[users] + self.bi[items] + np.sum(self.P[users] * self.Q[items], axis=1))

    def recommend(self, user_id, user_map, movie_map, df_movies, top_n=10):
        """
        Generate Top-N recommendations for a user.

        Parameters
        ---------------
        user_id : int
        user_map : dict
        movie_map : dict
        df_movies : pd.DataFrame
        top_n : int

        Returns
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

#======================================================================================================================
#=============================== Collaborative Filtering with Matrix Factorization (SVD)  ============================#
#======================================================================================================================
class MF_SVD:
    """
    Model-based Collaborative Filtering using Singular Value Decomposition (SVD).
    This implementation applies user-mean normalization.

    Parameters
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

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame containing columns ['u', 'i', 'rating'].
        n_users : int
            Total number of unique users.
        n_movies : int
            Total number of unique movies.

        Returns
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
            shape=(n_users, n_movies))

        # Perform Spectral Decomposition (SVD)
        u, s, vt = svds(train_sparse.astype(float), k=self.k)
        
        # Sort latent factors in descending order by singular values
        idx = np.argsort(s)[::-1]
        self.U = u[:, idx]
        self.sigma = np.diag(s[idx])
        self.Vt = vt[idx, :]

        return self

    def predict(self, user_indices, movie_indices):
        """
        Predict ratings for a batch of user-item pairs using matrix dot products.

        Parameters
        ----------
        user_indices : np.ndarray
            Array of mapped user indices ('u').
        movie_indices : np.ndarray
            Array of mapped movie indices ('i').

        Returns
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

        Parameters
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

        Returns
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


# def user_content_recommendations(user_id, ratings, movies, matrix, top_n=10, min_rating=3.5):
#     """
#     Generates movie recommendations for a given user based on their entire viewing history
#     using a content-based filtering approach.

#     The function builds a user profile by aggregating TF-IDF feature vectors of movies
#     that the user has positively rated (above a specified threshold). Each movie vector
#     is weighted by the user's rating, creating a representation of the user's preferences.
#     Then, cosine similarity is computed between the user profile and all movies to find
#     the most relevant recommendations.

#     Parameters
#     ----------
#     user_id : int
#         ID of the user for whom recommendations are generated.
#     ratings : pandas.DataFrame
#         DataFrame containing user ratings with columns:
#         - 'userId' : int
#         - 'movieId' : int
#         - 'rating' : float
#     movies : pandas.DataFrame
#         DataFrame containing movie metadata with columns:
#         - 'movieId' : int
#         - 'title' : str
#     matrix : scipy.sparse matrix
#         TF-IDF feature matrix where each row corresponds to a movie
#         (aligned with the `movies` DataFrame).
#     top_n : int, optional (default=10)
#         Number of recommendations to return.
#     min_rating : float, optional (default=3.5)
#         Minimum rating threshold used to determine which movies are considered
#         "liked" by the user.

#     Returns
#     -------
#     pandas.DataFrame
#         DataFrame containing the top-N recommended movies with columns:
#         - 'movieId'
#         - 'title'

#     Notes
#     -----
#     - Movies already rated by the user are excluded from recommendations.
#     - The function operates on sparse matrices for efficiency and avoids
#       converting them to dense format to reduce memory usage.
#     - If the user has no ratings above `min_rating`, an empty DataFrame is returned.
#     """

#     user_ratings = ratings[ratings["userId"] == user_id]
#     liked = user_ratings[user_ratings["rating"] >= min_rating]

#     if liked.empty:
#         return pd.DataFrame(columns=["movieId", "title"])

#     # movie indexes
#     movie_indices = movies.index[movies["movieId"].isin(liked["movieId"])]

#     # weights (ratings)
#     weights = liked.set_index("movieId").loc[
#         movies.loc[movie_indices, "movieId"]
#     ]["rating"].values

#     user_matrix = matrix[movie_indices]

#     user_profile = user_matrix.multiply(weights[:, np.newaxis]).sum(axis=0)

#     user_profile = np.asarray(user_profile)

#     # normalization
#     user_profile = user_profile / np.sum(weights)

#     # similarity
#     similarity = cosine_similarity(user_profile, matrix).flatten()

#     # filtering watched movies
#     seen_movies = set(user_ratings["movieId"])

#     indices = [
#         i for i in similarity.argsort()[::-1]
#         if movies.iloc[i]["movieId"] not in seen_movies
#     ][:top_n]

#     return movies.iloc[indices][["movieId", "title"]]

def user_content_recommendations(user_id, ratings, movies, matrix, top_n=10, min_rating=3.5):

    user_ratings = ratings[ratings["userId"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= min_rating]

    if liked.empty:
        return movies.iloc[[]][["movieId", "title"]]

    # indeksy filmów
    movie_indices = movies.index[movies["movieId"].isin(liked["movieId"])]

    weights = liked.set_index("movieId").loc[
        movies.loc[movie_indices, "movieId"]
    ]["rating"].values

    # profil użytkownika (sparse!)
    user_matrix = matrix[movie_indices]
    user_profile = user_matrix.multiply(weights[:, np.newaxis]).sum(axis=0)

    user_profile = np.asarray(user_profile)
    user_profile = normalize(user_profile)

    # 🔥 szybkie similarity
    similarity = user_profile @ matrix.T

    similarity = similarity.flatten()

    # 🔥 zamiast argsort → argpartition
    top_idx = np.argpartition(-similarity, top_n + 20)[:top_n + 20]

    # usuwamy obejrzane
    seen = set(user_ratings["movieId"])

    filtered = [
        i for i in top_idx
        if movies.iloc[i]["movieId"] not in seen
    ][:top_n]

    return movies.iloc[filtered][["movieId", "title"]]


#=================================================== EVALUATION CONTENT BASED ====================================================
def build_user_dicts(ratings_train, ratings_test):
    """
    Precomputes user data for fast lookup.
    """

    train_dict = {
        user: group
        for user, group in ratings_train.groupby("userId")
    }

    test_dict = {
        user: group[group["rating"] >= 3.5]["movieId"].tolist()
        for user, group in ratings_test.groupby("userId")
    }

    return train_dict, test_dict


def hit_rate_at_k(recommended, relevant, k=10):
    return int(len(set(recommended[:k]) & set(relevant)) > 0)


def evaluate_hit_rate_fast(model_func, ratings_train, ratings_test, movies, matrix, k=10):
    """
    Fast evaluation using Hit Rate@K with tqdm progress bar.
    """

    scores = []

    # 🔥 precompute
    train_dict, test_dict = build_user_dicts(ratings_train, ratings_test)

    users = list(test_dict.keys())

    for user_id in tqdm(users):

        relevant = test_dict[user_id]

        if len(relevant) == 0:
            continue

        recs = model_func(
            user_id=user_id,
            ratings=ratings_train,
            movies=movies,
            matrix=matrix,
            top_n=k
        )

        recommended = recs["movieId"].values

        score = hit_rate_at_k(recommended, relevant, k)
        scores.append(score)

    return sum(scores) / len(scores)




# def hit_rate_at_k(recommended, relevant, k=10):
#     """
#     Computes Hit Rate@K.
#     Returns 1 if at least one relevant item is in top-K recommendations, else 0.
#     """

#     recommended = recommended[:k]

#     return int(len(set(recommended) & set(relevant)) > 0)


def train_test_split_per_user(ratings, test_size=0.2):
    train_list = []
    test_list = []

    for user_id, group in ratings.groupby("userId"):
        if len(group) < 5:
            train_list.append(group)
            continue

        train, test = train_test_split(group, test_size=test_size, random_state=42)

        train_list.append(train)
        test_list.append(test)

    return pd.concat(train_list), pd.concat(test_list)


def evaluate_hit_rate(model_func, ratings_train, ratings_test, movies, matrix, k=10):
    """
    Evaluates recommender using Hit Rate@K.
    """

    scores = []

    for user_id in ratings_test["userId"].unique():

        # filmy które user naprawdę polubił (test)
        relevant = ratings_test[
            (ratings_test["userId"] == user_id) &
            (ratings_test["rating"] >= 3.5)
        ]["movieId"].tolist()

        if len(relevant) == 0:
            continue

        # rekomendacje
        recs = model_func(
            user_id=user_id,
            ratings=ratings_train,
            movies=movies,
            matrix=matrix,
            top_n=k
        )

        recommended = recs["movieId"].tolist()

        score = hit_rate_at_k(recommended, relevant, k)
        scores.append(score)

    return sum(scores) / len(scores)

#============================================== Hybrid model  ===========================================#

def content_based_scores(user_id, ratings, movies, matrix, min_rating=3.5):

    user_ratings = ratings[ratings["userId"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= min_rating]

    if liked.empty:
        return np.zeros(len(movies))

    movie_indices = movies.index[movies["movieId"].isin(liked["movieId"])]

    weights = liked.set_index("movieId").loc[
        movies.loc[movie_indices, "movieId"]
    ]["rating"].values

    user_matrix = matrix[movie_indices]
    user_profile = (user_matrix.multiply(weights[:, np.newaxis]).sum(axis=0))

    user_profile = normalize(np.asarray(user_profile))

    similarity = (user_profile @ matrix.T).flatten()

    return similarity

# SVD score 
# def cf_scores(model, user_id, user_map):

#     if user_id not in user_map:
#         return None

#     u = user_map[user_id]
#     u_mean = model.user_means.get(u, model.global_mean)

#     predictions = (u_mean + np.dot(np.dot(model.U[u], model.sigma), model.Vt))

#     return predictions

# SGD score
def cf_scores(model, user_id, user_map):

    if user_id not in user_map:
        return None

    u = user_map[user_id]

    predictions = (model.mu + model.bu[u] + model.bi + model.P[u].dot(model.Q.T))

    return predictions

def hybrid_model_recommendations(user_id, ratings, movies, matrix, cf_model, user_map, movie_map, alpha, top_n):
    
    cb_scores = content_based_scores(user_id, ratings, movies, matrix)
    cf_scores_results = cf_scores(cf_model, user_id, user_map)
    
    if cf_scores_results is None:
        return user_content_recommendations(user_id, ratings, movies, matrix, top_n)
    
    cb_scores = MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
    cf_scores = MinMaxScaler().fit_transform(cf_scores_results.reshape(-1, 1)).flatten()
    
    hybrid_scores = alpha*cf_scores_results + (1 - alpha)*cb_scores
    
    movies_seen = set(ratings[ratings["userId"] == user_id]["movieId"])
    recommendations_idx = []
    
    for index in np.argsort(hybrid_scores)[::-1]:
        
        movie_id = movies.iloc[index]["movieId"]
        if movie_id not in movies_seen:
            recommendations_idx.append(index)
            
        if len(recommendations_idx) == top_n:
            break
        
    return movies.iloc[recommendations_idx][["movieId", "title"]]