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
# from tqdm.notebook import tqdm
from tqdm import tqdm
import time
import os


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

    # movie indexes
    movie_indices = movies.index[movies["movieId"].isin(liked["movieId"])]

    weights = liked.set_index("movieId").loc[
        movies.loc[movie_indices, "movieId"]
    ]["rating"].values

    # user profile (sparse)
    user_matrix = matrix[movie_indices]
    user_profile = user_matrix.multiply(weights[:, np.newaxis]).sum(axis=0)

    user_profile = np.asarray(user_profile)
    user_profile = normalize(user_profile)

    # similarity
    similarity = user_profile @ matrix.T

    similarity = similarity.flatten()

    # argpartition
    top_idx = np.argpartition(-similarity, top_n + 20)[:top_n + 20]

    # we're removing watched movies from the top candidates
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

    # precompute
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

        # movies that the user really liked (test)
        relevant = ratings_test[
            (ratings_test["userId"] == user_id) &
            (ratings_test["rating"] >= 3.5)
        ]["movieId"].tolist()

        if len(relevant) == 0:
            continue

        # recommendations
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
def get_cf_scores(model, user_id, user_map, movie_map, movies):

    if user_id not in user_map:
        return None

    u = user_map[user_id]

    predictions = (model.mu + model.bu[u] + model.bi + model.P[u].dot(model.Q.T))
    
    full_predictions = np.zeros(len(movies))
    inv_movie_map = {v: k for k, v in movie_map.items()}
    movie_index_map = {movie_id: index for index, movie_id in enumerate(movies["movieId"])}

    for mf_index, score in enumerate(predictions):

        movie_id = inv_movie_map[mf_index]

        if movie_id in movie_index_map:
            full_predictions[movie_index_map[movie_id]] = score

    return full_predictions

def hybrid_model_recommendations(user_id, ratings, movies, matrix, cf_model, user_map, movie_map, alpha, top_n):
    
    cb_scores = content_based_scores(user_id, ratings, movies, matrix)
    cf_scores_results = get_cf_scores(cf_model, user_id, user_map, movie_map, movies)
    
    if cf_scores_results is None:
        return user_content_recommendations(user_id, ratings, movies, matrix, top_n)
    
    cb_scores = MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
    cf_scores_results = MinMaxScaler().fit_transform(cf_scores_results.reshape(-1, 1)).flatten()
    
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

#======================================================================================================================
#=================================================== COLD START =======================================================
#======================================================================================================================
def cold_start_flow(
    ratings,
    movies,
    movies_plus_tags,
    matrix,
    content_based_recommendations,
    user_content_recommendations,
    ratings_path
):
    """
    Performs an interactive cold-start procedure for a new user and generates
    movie recommendations based on the provided ratings.

    The function presents a predefined list of popular movies and asks the user
    to rate each movie on a scale from 0 to 5 (including half-point ratings).
    Based on the collected ratings, a new user profile is created and stored
    in the ratings dataset.

    The workflow consists of the following steps:

    1. Display a list of predefined movies and collect user ratings.
    2. Validate the entered ratings and allow one retry for invalid input.
    3. Create a new unique user identifier.
    4. Save all non-zero ratings to the ratings dataset.
    5. Generate content-based recommendations for each positively rated movie
       (rating >= 3.5).
    6. If the user does not provide any positive ratings, use a predefined
       fallback set of popular movies.
    7. Generate personalized recommendations using the user's complete rating
       profile through the user-based content recommendation model.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Ratings dataset containing at least the following columns:
        'userId', 'movieId', 'rating', and 'timestamp'.

    movies : pandas.DataFrame
        Movie metadata containing at least 'movieId' and 'title' columns.

    movies_plus_tags : pandas.DataFrame
        Movie dataset enriched with tags and content features used by the
        recommendation system.

    matrix : scipy.sparse.csr_matrix
        TF-IDF feature matrix generated from movie genres and tags.

    content_based_recommendations : callable
        Function used to generate recommendations similar to a given movie.
        Expected signature:

            content_based_recommendations(
                movie_id,
                df,
                matrix,
                top_n=10
            )

    user_content_recommendations : callable
        Function used to generate personalized recommendations based on a
        user's rating history. Expected signature:

            user_content_recommendations(
                user_id,
                ratings,
                movies,
                matrix,
                top_n=10
            )

    ratings_path : str or pathlib.Path
        Path to the ratings CSV file where the new user's ratings will be
        stored.

    Returns
    -------
    int
        Identifier of the newly created user.

    Notes
    -----
    Only ratings greater than zero are stored in the ratings dataset.

    Ratings equal to or greater than 3.5 are treated as positive preferences
    and are used to generate content-based recommendations.

    If the user does not provide any positive ratings, a predefined set of
    popular movies is used to initialize the recommendation process.
    """
    
    movies_to_rate = {
        1: "Toy Story (1995)",
        2: "Jumanji (1995)",
        10: "GoldenEye (1995)",
        318: "Shawshank Redemption, The (1994)",
        296: "Pulp Fiction (1994)",
        356: "Forrest Gump (1994)",
        527: "Schindler's List (1993)",
        3147: "Green Mile, The (1999)",
        480: "Jurassic Park (1993)",
        1721: "Titanic (1997)",
        2571: "Matrix, The (1999)",
        72998: "Avatar (2009)",
        3578: "Gladiator (2000)",
        130219: "The Dark Knight (2011)"
    }

    allowed_ratings = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    print("""
    Did you like this movie?

    1 = strongly dislike
    2 = dislike
    3 = neutral
    4 = like
    5 = strongly like
    0 = not seen

    You can also use half ratings (e.g. 1.5, 2.5, 3.5, 4.5)
    """)

    ratings_cold_start = {}

    # -------------------------
    # 1. COLLECT RATINGS
    # -------------------------
    for movie_id, title in movies_to_rate.items():

        attempts = 0

        while True:
            try:
                rating = float(input(f"Rate '{title}' (0-5): "))

                if rating in allowed_ratings:
                    ratings_cold_start[movie_id] = rating
                    break

                attempts += 1

                if attempts >= 2:
                    print("I see you can't make up your mind, so let's move on...")
                    break

                print("Invalid rating. Try again (0,1,1.5,...,5)")

            except:
                attempts += 1
                print("Invalid input. Try again.")

                if attempts >= 2:
                    print("I see you can't make up your mind, so let's move on...")
                    break

    # -------------------------
    # 2. CREATE NEW USER ID
    # -------------------------
    target_user_id = ratings["userId"].max() + 1

    # -------------------------
    # 3. SAVE RATINGS
    # -------------------------
    new_rows = []

    timestamp = int(time.time())

    for movie_id, rating in ratings_cold_start.items():
        if rating != 0:
            new_rows.append({
                "userId": target_user_id,
                "movieId": movie_id,
                "rating": rating,
                "timestamp": timestamp
            })

    # append to CSV
    if len(new_rows) > 0:

        new_df = pd.DataFrame(new_rows)

        if os.path.exists(ratings_path):
            existing = pd.read_csv(ratings_path)
            updated = pd.concat([existing, new_df], ignore_index=True)
        else:
            updated = new_df

        updated.to_csv(ratings_path, index=False)

        print(f"\nSaved {len(new_rows)} ratings for new user.")

    # -------------------------
    # 4. CONTENT-BASED RECOMMENDATIONS
    # -------------------------
    liked_movies = [
        mid for mid, r in ratings_cold_start.items() if r >= 3.5
    ]

    if len(liked_movies) == 0:

        print("\nUser gave no meaningful ratings => fallback popular movies")

        liked_movies = [6377, 59315, 49272, 2959, 4995]

        # fallback ratings
        ratings_cold_start.update({
            6377: 5, #Finding Nemo (2003)
            59315: 4.5, #Iron Man (2008)
            49272: 4, #Casino Royale (2006)
            2959: 4.5, #Fight Club (1999)
            4995: 3.5 #Beautiful Mind, A (2001)
        })

    print("\n--- CONTENT-BASED RECOMMENDATIONS ---\n")

    for mid in liked_movies:

        title = movies[movies["movieId"] == mid]["title"].values[0]

        print(f"Since you like the movie: {title}")

        recs = content_based_recommendations(
            movie_id=mid,
            df=movies_plus_tags,
            matrix=matrix,
            top_n=10
        )

        print(recs)
        print("\n" + "-" * 50 + "\n")

    # -------------------------
    # 5. USER-BASED RECOMMENDATIONS
    # -------------------------
    print("\n--- USER-BASED RECOMMENDATIONS ---\n")
    ratings = pd.read_csv(ratings_path)

    recs_user = user_content_recommendations(
        user_id=target_user_id,
        ratings=ratings,
        movies=movies_plus_tags,
        matrix=matrix,
        top_n=10
    )

    print(recs_user)

    return target_user_id