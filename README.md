# Movie Recommendation System

## Overview

This project implements and compares multiple recommendation approaches using the MovieLens dataset.

The goal is to recommend movies to users based on their historical ratings and movie content. The project explores three recommendation strategies:

* Collaborative Filtering
* Content-Based Filtering
* Hybrid Recommendation System

Additionally, a solution to the **Cold Start Problem** is implemented to provide recommendations for new users with no prior rating history.

---

## Dataset

The project uses the **MovieLens** dataset, which contains:

### Movies

* `movieId`
* `title`
* `genres`

### Ratings

* `userId`
* `movieId`
* `rating`
* `timestamp`

### Tags

* `userId`
* `movieId`
* `tag`
* `timestamp`

MovieLens provides a large collection of user ratings and movie metadata, making it a popular benchmark for recommender systems research.

---

## Repository Structure

```text
my-ml-project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 00_prepocessing.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_interpretability.ipynb
│   └── 04_final_report.ipynb
├── src/
│   ├── data_loader.py
|   ├── explainability_plots.py
|   ├── explainability.py
│   ├── features.py
│   └── models.py
├── .gitignore
├── mlartifacts/ 
├── mlflow.db
├── README.md
└── requirements.txt
```

---

## Recommendation Methods

### 1. Collaborative Filtering

Collaborative Filtering predicts user preferences based on historical user–item interactions.

Implemented approaches:

* Singular Value Decomposition (SVD)
* Matrix Factorization trained using Stochastic Gradient Descent (SGD)

The model learns latent factors representing users and movies and predicts ratings for unseen items.

Evaluation metrics:

* RMSE
* MAE

---

### 2. Content-Based Filtering

Content-Based Filtering recommends movies based on movie characteristics.

Movie representations are built using:

* Genres
* Tags

Genres and tags are transformed into TF-IDF vectors, and movie similarity is calculated using cosine similarity.

Implemented recommendation modes:

* Movie-to-movie recommendations
* User profile recommendations

Evaluation metric:

* Hit Rate@10

---

### 3. Hybrid Recommendation System

The Hybrid Recommendation System combines Collaborative Filtering and Content-Based Filtering.

The goal is to leverage:

* Collaborative Filtering's ability to learn user behaviour patterns.
* Content-Based Filtering's ability to utilize movie metadata.

The hybrid approach aims to improve recommendation quality by combining both sources of information.

---

## Cold Start Problem

One of the main challenges in recommender systems is the cold start problem.

When a new user joins the platform, there is no historical data available to generate personalized recommendations.

To address this issue, an interactive onboarding procedure was implemented:

1. The user rates a predefined set of popular movies.
2. An initial user profile is created.
3. Content-based recommendations are generated.
4. Personalized recommendations are produced immediately after registration.

This allows the system to recommend movies even for users without any previous interaction history.

---

## Running the Project

The recommended workflow is:

0. `00_preprocessing.ipynb` – Data Preprocessing
1. `01_eda.ipynb` – Exploratory Data Analysis
3. `02_modeling.ipynb` – Implementation, demonstration and evaluation of recommendation methods
4. `03_interpretability.ipynb` – SHAP interpretation and explainability
5. `04_final_report.ipynb` – Final evaluation and results

---

## Results

The project evaluates recommendation quality using:

### Collaborative Filtering

* RMSE
* MAE

### Content-Based Filtering

* Hit Rate@10

### Hybrid Model

* Hit Rate@10

### Cold Start

* Qualitative evaluation of generated recommendations

---

## Future Improvements

Potential extensions include:

* Deep Learning-based recommender systems
* Deployment as a web application

---

## Authors

Project developed by:

- Jagoda Płócienniczak
- Paweł Stępień
- Magdalena Sudół
- Jan Zawadzki

Course: Machine Learning for Data Analytics 
University: Politechnika Wrocławska  
Academic Year: 2025/2026
