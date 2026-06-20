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
    The class explains hybrid recommendations using Random Forest surrogate model and SHAP library.
    It chooses only movie genres and top 100 tags in order to optimize the performance.
    """
    def __init__(self, user_id, ratings, movies, matrix, vectorizer, cf_model, user_map, movie_map, alpha=0.5, mode='hybrid'):
        """
        The method initializes the RecommenderExplainer.
        
        PARAMETERS:
        - user_id: ID of the user to explain;
        - ratings: a dataframe with user's ratings;
        - movies: a dataframe with movie metadata;
        - matrix: sparse matrix of movie features;
        - vectorizer: TfidVectorizer used to build the content matrix;
        - cf_model: trained CF (Collaborative Filtering) model;
        - user_map: mapping of user's ID;
        - alpha: the weight of hybrid model (0 < alpha < 1);
        - mode: model to choose, "cf" (Collaborative Filtering), "cb" (Content-Based) or "hybrid" (Hybrid).
        """
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

        # we map models to names that will be used on plots
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
        Generelized method to retrieve results of different models (Content-based, Collaborative Filtering, Hybrid). The method normalizes scores based on the selected mode.

        PARAMETERS: None.
        
        RETURNS: 
        - NumPy array of normalized scores.
        """
        cb_scores = content_based_scores(self.user_id, self.ratings, self.movies, self.matrix)
        cf_scores = get_cf_scores(self.cf_model, self.user_id, self.user_map, self.movie_map, self.movies)
        
        # choosing Content-Based model
        if self.mode == 'cb':
            return MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
        
        # choosing Collaborative Filtering model
        elif self.mode == 'cf':
            if cf_scores is None: raise ValueError("CF model not available")
            return MinMaxScaler().fit_transform(cf_scores.reshape(-1, 1)).flatten()
        
        # choosing Hybrid model (default)
        else: 
            if cf_scores is None: return MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
            cb_scaled = MinMaxScaler().fit_transform(cb_scores.reshape(-1, 1)).flatten()
            cf_scaled = MinMaxScaler().fit_transform(cf_scores.reshape(-1, 1)).flatten()
            return self.alpha * cf_scaled + (1 - self.alpha) * cb_scaled

    def _select_top_features(self):
        """
        The method identifies genres' indices and selects top 100 tags. The method ignores noise (numbers, stop-words or technical tags etc.).

        PARAMETERS: None.

        RETURNS: None.
        """
        # we define tokens corresponding to movies' genres from MovieLens
        genres_list = {
            'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 
            'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 
            'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western', 'imax'
        }
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # we take indices of the genres
        genre_indices = [i for i, name in enumerate(feature_names) if name in genres_list]
        
        # we take indices of important words and reject the junk words (numbers, stop-words etc.)
        tag_indices = [i for i, name in enumerate(feature_names) if name not in genres_list]
        
        # dictionary storing junk words from MovieLens database
        junk_words = {'owned', 'dvd', 'on', 'avi', 'seen', 'buy', 'queue', 'netflix', 'imdb', 'top', 'see', 'the'}
        
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
        The method trains Random Forest surrogate model to mimic the complex recommendation engine.

        PARAMETERS: 
        - sample_size: number of samples used to train the surrogate Random Forest model.

        RETURNS: None.
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
        
        # we evaluate the explanation with r2 metrics
        y_pred = self.surrogate_model.predict(self.X_df)
        r2 = r2_score(y_sample, y_pred)
        
        # we print the results retrieved
        print(f"Explainability metrics for User {self.user_id}")
        print(f"Analysed features: {len(self.selected_feature_names)} (all genres + top 100 tags)")
        print(f"Surrogate model credibility (R^2 Score): {r2:.4f}")
        
        # we calculate SHAP value for global (user's preference) and local (recommendation explanation) explanation
        explainer = shap.TreeExplainer(self.surrogate_model)
        self.shap_values = explainer(self.X_df)

    def plot_global_explanation(self, max_display=15):
        """
        The method creates beeswarm plots for a given user and shows the plot.

        PARAMETERS: 
        - max_display: maximum number of features presented on the beeswarm plot.
        
        RETURNS: None.
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
            The method creates waterfall plot explaining why a specific movie was recommended to the user and shows the plot.

            PARAMETERS:
            - movie_title: database title of the recommended movie.

            RETURNS: None.

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

    def plot_summary_importance(self, max_display=15):
            """
            The method creates a SHAP summary bar plot showing global feature importance.

            PARAMETERS:
            - max_display: maximum number of features presented on the bar plot.
            
            RETURNS: None.
            """
            if self.shap_values is None: 
                raise ValueError("Start the prepare_surrogate() method first!")
                
            plt.figure(figsize=(10, 6))
            plt.title(f"Global feature importance - summary plot (model: {self.mode_name})", fontsize=14, pad=20)
            shap.plots.bar(self.shap_values, max_display=max_display, show=False)
            plt.tight_layout()
            plt.show()

    def plot_dependence(self, feature_name):
        """
        The method creates a SHAP dependence plot to visualize how a single feature 
        affects predictions and its interaction with other automatically selected features.

        PARAMETERS:
        - feature_name: string name of the feature to analyze (e.g., 'genres_Drama' or 'atmospheric').
        
        RETURNS: None.
        """
        if self.shap_values is None: 
            raise ValueError("Start the prepare_surrogate() method first!")
            
        if feature_name not in self.selected_feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in analyzed features.")
            
        shap.plots.scatter(self.shap_values[:, feature_name], color=self.shap_values, show=False)
        plt.title(f"SHAP Dependence Plot for '{feature_name}' (model: {self.mode_name})", fontsize=12, pad=15)
        plt.gcf().set_size_inches(8, 5) # Bezpieczne wymuszenie rozmiaru (8x5 cali) na aktywnym wykresie
        
        plt.tight_layout()
        plt.show()

    def explain_counterfactual(self, movie_title, top_n=5):
        """
        The method performs a "What-If" counterfactual analysis for a specific movie.
        It simulates toggling the movie's features (adding missing tags or removing existing ones) 
        to show how the recommendation score would change.

        PARAMETERS:
        - movie_title: database title of the movie;
        - top_n: number of top counterfactual impacts to display.
        
        RETURNS: None.
        """
        if self.surrogate_model is None:
            raise ValueError("Start the prepare_surrogate() method first!")
            
        movie_row = self.movies[self.movies['title'].str.contains(movie_title, case=False, na=False, regex=False)]
        if movie_row.empty:
            print(f"'{movie_title}' was not found!")
            return
            
        orig_idx = movie_row.index[0]
        actual_title = movie_row.iloc[0]['title']
        
        # Extract original feature vector for this movie
        X_movie = self.matrix[orig_idx][:, self.selected_indices].toarray()
        movie_df = pd.DataFrame(X_movie, columns=self.selected_feature_names)
        
        # Get baseline prediction from surrogate model
        orig_pred = self.surrogate_model.predict(movie_df)[0]
        
        print(f"========================================================")
        print(f"  Counterfactual analysis for: '{actual_title}'")
        print(f"  Original recommendation evaluation (surogate model): {orig_pred:.4f}")
        print("========================================================")
        
        counterfactuals = []
        
        # Test what happens if we toggle each feature individually
        for col in self.selected_feature_names:
            current_val = movie_df.loc[0, col]
            perturbed_df = movie_df.copy()
            
            if current_val > 0:
                # feature is present. What if it was ABSENT?
                perturbed_df.loc[0, col] = 0.0
                new_pred = self.surrogate_model.predict(perturbed_df)[0]
                diff = new_pred - orig_pred
                counterfactuals.append({
                    'feature': col,
                    'action': 'REMOVING FEATURE',
                    'diff': diff,
                    'new_score': new_pred
                })
            else:
                # feature is absent. What if it was PRESENT? (Simulated as TF-IDF 1.0)
                perturbed_df.loc[0, col] = 1.0
                new_pred = self.surrogate_model.predict(perturbed_df)[0]
                diff = new_pred - orig_pred
                counterfactuals.append({
                    'feature': col,
                    'action': 'ADDING FEATURE',
                    'diff': diff,
                    'new_score': new_pred
                })
                
        cf_df = pd.DataFrame(counterfactuals)
        
        # what would lower the recommendation if it was absent?
        print(f"\n[CRUCIAL FEATURES] Which of the following characteristics would most LOWER the recommendation?:")
        neg_effects = cf_df[cf_df['diff'] < 0].sort_values(by='diff').head(top_n)
        if neg_effects.empty:
            print(" - No traits with negative effects after removal.")
        for _, row in neg_effects.iterrows():
            print(f" - If this movie LOST feature '{row['feature']}': The rating is lower by: {row['diff']:.4f} -> New result: {row['new_score']:.4f}")
            
        # what would raise the recommendation if it was present?
        print(f"\nWhat can be added/changed to make user like the movie more?:")
        pos_effects = cf_df[cf_df['diff'] > 0].sort_values(by='diff', ascending=False).head(top_n)
        if pos_effects.empty:
            print(" - This movie is already perfect for the user!")
        for _, row in pos_effects.iterrows():
            print(f" - If this movie HAD feature '{row['feature']}': The rating is higer by {row['diff']:.4f} -> New result: {row['new_score']:.4f}")
        print("="*70)