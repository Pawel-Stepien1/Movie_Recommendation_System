#======================================================================================================================
#====================================================== IMPORT ========================================================
#======================================================================================================================

from data_loader import read_csv_files, cleaning_raw_data
from features import merge_movies_with_tags, build_content_matrix, prepare_cf_data, cf_to_numpy
from models import MF_SGD, hybrid_model_recommendations
from explainability import RecommenderExplainer
import pandas as pd

#======================================================================================================================
#============================================== PREPARING THE DATA ====================================================
#======================================================================================================================

# we prepare the data
links, movies, ratings, tags = read_csv_files()
links, movies, ratings, tags = cleaning_raw_data(links, movies, ratings, tags)
merged_movies = merge_movies_with_tags(movies, tags)
matrix, vectorizer = build_content_matrix(merged_movies)

train_df, test_df, user_map, movie_map = prepare_cf_data(ratings)
train_users, train_items, train_ratings, _, _, _ = cf_to_numpy(train_df, test_df)

# training CF model
n_users = len(user_map)
n_items = len(movie_map)
sgd_model = MF_SGD(k=20, epochs=5)  # 5 epochs for testing
sgd_model.fit(train_users, train_items, train_ratings, n_users, n_items)

# ======================================================================================================================
# =========================================== EXPLAINABILITY CLASS CALL ================================================
# ======================================================================================================================

# we select random user 
target_user = list(user_map.keys())[0]

# we choose the best movie for a user
chosen_recommendation = hybrid_model_recommendations(
    user_id=target_user,
    ratings=train_df,
    movies=merged_movies,
    matrix=matrix,
    cf_model=sgd_model,
    user_map=user_map,
    movie_map=movie_map,
    alpha=0.4,
    top_n=5
)

# we choose the movie rated as the best (number 1) in recommendation ranking;
# the code is adapted to whether we use a DataFrame or a list
if isinstance(chosen_recommendation, pd.DataFrame):
    best_engine_movie = chosen_recommendation.iloc[0]['title']
else:
    # in case of a list
    best_engine_movie = chosen_recommendation[0]

print(f"\nEngine recommends {best_engine_movie} for user {target_user}.")

# we initialize the explainer
explainer = RecommenderExplainer(
    user_id=target_user,
    ratings=train_df,
    movies=merged_movies,
    matrix=matrix,
    vectorizer=vectorizer,
    cf_model=sgd_model,
    user_map=user_map,
    movie_map=movie_map,
    alpha=0.4 # weight of Collaborative Filtering model
)

# we match the interpretable surrogate model
explainer.prepare_surrogate(sample_size=400)

# we create beeswarm plot (user preference profile)
# the plots explains which words/genres raise the user's rating and which words/genres lower the user's rating
explainer.plot_global_explanation(max_display=12)

# we create waterfall plot (user preference profile)
explainer.explain_specific_movie(best_engine_movie)