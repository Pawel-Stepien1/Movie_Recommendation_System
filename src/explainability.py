#======================================================================================================================
#====================================================== IMPORT ========================================================
#======================================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from models import content_based_scores, get_cf_scores

#======================================================================================================================
#================================================ EXPLAINER CLASS =====================================================
#======================================================================================================================

class RecommenderExplainer:
    """
    The class explains hybrid recommendations using Surrogate Model and SHAP library.
    It chooses only movie genres and top 100 tags in order to optimize the performance.
    """
    def __init__(self, user_id, ratings, movies, matrix, vectorizer, cf_model, user_map, movie_map, alpha=0.5, mode='hybrid'):
        self.user_id = user_id
        self.ratings = ratings
        self.movies = movies.reset_index(drop=True)
        self.matrix = matrix
        self.vectorizer = vectorizer
        self.cf_model = cf_model
        self.user_map = user_map
        self.movie_map = movie_map
        self.alpha = alpha
        self.mode = mode
        self.mode_name = {"cb" : "Content-Based", "cf" : "Collaborative Filtering", "hybrid" : "Hybrid"}.get(mode, "Model")
        
        self.surrogate_model = None
        self.shap_values = None
        self.X_df = None
        self.sample_indices = None
        
        # atributes storing selected features (genres and top 100 tags)
        self.selected_indices = None
        self.selected_feature_names = None

    def _get_scores(self):
        """
        Generelized method to get results of different models (Content-based, Collaborative Filtering, Hybrid).
        """
        cb_scores = content_based_scores(self.user_id, self.ratings, self.movies, self.matrix)
        cf_scores = get_cf_scores(self.cf_model, self.user_id, self.user_map, self.movie_map, self.movies)
        
        if self.mode == 'cb':
            return MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
        
        elif self.mode == 'cf':
            if cf_scores is None: raise ValueError("CF model not available")
            return MinMaxScaler().fit_transform(cf_scores.reshape(-1, 1)).flatten()
        
        else: # Tryb hybrydowy (domyślny)
            if cf_scores is None: return MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
            cb_scaled = MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
            cf_scaled = MinMaxScaler().fit_transform(cf_scores.reshape(-1, 1)).flatten()
            return self.alpha * cf_scaled + (1 - self.alpha) * cb_scaled

    def _select_top_features(self):
        """
        The function identifies genres' indices and selects top 100 tags. 
        It ignores noise (numbers, stop-words or technical tags etc.). 
        """
        # we define tokens corresponding to movies' genres from MovieLens
        genres_list = {
            'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 
            'documentary', 'drama', 'fantasy', 'film', 'noir', 'horror', 
            'musical', 'mystery', 'romance', 'sci', 'fi', 'thriller', 'war', 'western', 'imax'
        }
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # we take indices of the genres
        genre_indices = [i for i, name in enumerate(feature_names) if name in genres_list]
        
        # we take indices of important words and reject the junk words (numbers, stop-words etc.)
        tag_indices = [i for i, name in enumerate(feature_names) if name not in genres_list]
        
        # dictionary storing junk words from MovieLens database
        junk_words = {'owned', 'dvd', 'on', 'avi', 'seen', 'buy', 'queue', 'netflix', 'imdb', 'top', 'see'}
        
        valid_tag_indices = []
        for idx in tag_indices:
            name = feature_names[idx]
            # we consider numbers, words shorter than 3 characters (e.g. "on", "at") and words from junk_words as invalid tags
            if name.isdigit() or len(name) <= 2 or name in junk_words:
                continue
            valid_tag_indices.append(idx)
            
        # we select top 100 tags based on IDF (Inverse Document Frequency) index
        idfs = self.vectorizer.idf_
        valid_tag_idfs = idfs[valid_tag_indices]
        
        top_100_sub_indices = np.argsort(valid_tag_idfs)[:100]
        top_100_tag_indices = [valid_tag_indices[i] for i in top_100_sub_indices]
        
        # we combine genres and top 100 tags into one list of indices
        self.selected_indices = np.concatenate([genre_indices, top_100_tag_indices])
        self.selected_feature_names = feature_names[self.selected_indices]

    def prepare_surrogate(self, sample_size=300):
        """
        The function creates surrogate Random Forest model.
        """
        # we select the data
        if self.selected_indices is None:
            self._select_top_features()
            
        used_scored = self._get_scores()
        
        # we select the best rated movies + random sample in order to balance the training
        top_indices = np.argsort(used_scored)[::-1][:50]
        random_indices = np.random.choice(np.arange(len(self.movies)), size=min(sample_size, len(self.movies)), replace=False)
        self.sample_indices = np.unique(np.concatenate([top_indices, random_indices]))
        
        # we create feature matrix using only selected columns (genres and top 100 tags)
        X_sample = self.matrix[self.sample_indices][:, self.selected_indices].toarray()
        y_sample = used_scored[self.sample_indices]
        
        self.X_df = pd.DataFrame(X_sample, columns=self.selected_feature_names)
        
        # Random Forest training
        self.surrogate_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.surrogate_model.fit(self.X_df, y_sample)
        
        # we evaluate the exlanation with r2 metrics
        y_pred = self.surrogate_model.predict(self.X_df)
        r2 = r2_score(y_sample, y_pred)
        
        print(f"Explainability metrics for User {self.user_id}")
        print(f"Analysed features: {len(self.selected_feature_names)} (all genres + top 100 tags)")
        print(f"Surrogate model credibility (R^2 Score): {r2:.4f}")
        
        # we calculate SHAP value
        explainer = shap.TreeExplainer(self.surrogate_model)
        self.shap_values = explainer(self.X_df)

    def plot_global_explanation(self, max_display=15):
        """
        The function creates beeswarm plots for a given user.
        """
        if self.shap_values is None: raise ValueError("Start prepare_surrogate() first!")
        plt.figure(figsize=(10, 6))
        # Użycie self.mode_name w tytule
        plt.title(f"User {self.user_id} preference profile (model: {self.mode_name})", fontsize=14, pad=20)
        shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()

    def explain_specific_movie(self, movie_title):
            """
            The function explains why a specific movie was choses using waterfall plots.
            """
            if self.shap_values is None:
                raise ValueError("Start the prepare_surrogate() method first!")
                
            
            movie_row = self.movies[self.movies['title'].str.contains(movie_title, case=False, na=False, regex=False)]
            
            if movie_row.empty:
                print(f"'{movie_title}' was not found!")
                return
                
            orig_idx = movie_row.index[0]
            actual_title = movie_row.iloc[0]['title']
            
            if orig_idx not in self.sample_indices:
                X_movie = self.matrix[orig_idx][:, self.selected_indices].toarray()
                movie_df = pd.DataFrame(X_movie, columns=self.selected_feature_names)
                explainer = shap.TreeExplainer(self.surrogate_model)
                movie_shap = explainer(movie_df)
                
                plt.figure(figsize=(12, 5))
                plt.title(f"Why {actual_title} is recommended? (model: {self.mode_name})", fontsize=12, pad=20)
                shap.plots.waterfall(movie_shap[0], max_display=10, show=False)
                plt.tight_layout()
                plt.show()
            else:
                sample_idx = np.where(self.sample_indices == orig_idx)[0][0]
                
                plt.figure(figsize=(12, 5))
                plt.title(f"Why {actual_title} is recommended? (model: {self.mode_name})", fontsize=12, pad=20)
                shap.plots.waterfall(self.shap_values[sample_idx], max_display=10, show=False)
                plt.tight_layout()
                plt.show()