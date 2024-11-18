import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load data from CSV files
filtered_books_file_path = 'newData\\Filtered_Books.csv'
filtered_ratings_file_path = 'newData\\Filtered_Ratings.csv'
filtered_users_file_path = 'newData\\Filtered_Users.csv'
genres_file_path = 'newData\\Genres.csv'

books = pd.read_csv(filtered_books_file_path)  # Book details
ratings = pd.read_csv(filtered_ratings_file_path)  # User ratings for books
users = pd.read_csv(filtered_users_file_path)  # User details
genres = pd.read_csv(genres_file_path)  # Book genres

# 1. Data preprocessing
# Create a user-item ratings matrix, where rows represent users, columns represent items (books), and values are ratings.
ratings_matrix = ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
# Transpose of the ratings matrix for item-based collaborative filtering
ratings_matrix_T = ratings_matrix.T

# Function to calculate similarity using a specified metric
def calculate_similarity(matrix, metric='cosine'):
    if metric == 'cosine':
        return cosine_similarity(matrix)  # Cosine similarity
    elif metric == 'euclidean':
        return 1 / (1 + euclidean_distances(matrix))  # Convert Euclidean distances to similarity
    elif metric == 'pearson':
        return np.corrcoef(matrix)  # Pearson correlation
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")  # Raise an error for invalid metrics

# User-User Collaborative Filtering function
def user_based_cf(user_id, n_recommendations=5, metric='cosine'):
    # Compute user similarity matrix
    user_similarity = calculate_similarity(ratings_matrix, metric=metric)
    user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
    
    # Sort similar users for the target user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    
    # Aggregate ratings from similar users weighted by similarity scores
    user_ratings = ratings_matrix.loc[similar_users.index]
    scores = (user_ratings.T * similar_users).T.sum(axis=0) / similar_users.sum()
    
    # Filter out already-rated items
    already_rated = ratings_matrix.loc[user_id] > 0
    recommended_items = scores[~already_rated].sort_values(ascending=False).head(n_recommendations)
    return recommended_items.index.tolist()

# Item-Item Collaborative Filtering function
def item_based_cf(user_id, n_recommendations=5, metric='cosine'):
    # Compute item similarity matrix
    item_similarity = calculate_similarity(ratings_matrix_T, metric=metric)
    item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)
    
    # Calculate scores based on items rated by the user
    user_ratings = ratings_matrix.loc[user_id]
    scores = user_ratings.dot(item_similarity_df) / item_similarity_df.sum(axis=1)
    
    # Filter out already-rated items
    recommended_items = scores[user_ratings == 0].sort_values(ascending=False).head(n_recommendations)
    return recommended_items.index.tolist()

# Model-Based Collaborative Filtering function using SVD
def model_based_cf(user_id, n_recommendations=5):
    # Prepare data for the Surprise library
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    
    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
    # Train an SVD model
    algo = SVD()
    algo.fit(trainset)
    
    # Predict ratings for items not rated by the user
    user_ratings = ratings_matrix.loc[user_id]
    predictions = [
        (book, algo.predict(user_id, book).est)
        for book in ratings_matrix.columns if user_ratings[book] == 0
    ]
    
    # Sort predictions by estimated ratings
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_items = [isbn for isbn, _ in predictions[:n_recommendations]]
    return recommended_items

# Ensemble recommendation system combining user-based, item-based, and model-based approaches
def ensemble_recommendations(user_id, n_recommendations=5):
    # Generate recommendations from each method
    user_based_recommendations = user_based_cf(user_id, n_recommendations * 2)
    item_based_recommendations = item_based_cf(user_id, n_recommendations * 2)
    model_based_recommendations = model_based_cf(user_id, n_recommendations * 2)
    
    # Weight each method based on the size of the dataset
    total_ratings = len(ratings)
    if total_ratings > 10000:
        model_weight = 0.6
        user_weight = 0.2
        item_weight = 0.2
    else:
        model_weight = 0.2
        user_weight = 0.4
        item_weight = 0.4

    # Function to assign scores to recommendations based on weight
    def score_recommendations(recommendations, weight):
        return {isbn: weight for isbn in recommendations}

    # Combine scores from all methods
    user_scores = score_recommendations(user_based_recommendations, user_weight)
    item_scores = score_recommendations(item_based_recommendations, item_weight)
    model_scores = score_recommendations(model_based_recommendations, model_weight)

    combined_scores = {}
    for rec_dict in [user_scores, item_scores, model_scores]:
        for isbn, score in rec_dict.items():
            if isbn in combined_scores:
                combined_scores[isbn] += score
            else:
                combined_scores[isbn] = score

    # Sort recommendations by combined score and select top N
    sorted_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    final_recommendations = [isbn for isbn, score in sorted_recommendations[:n_recommendations]]
    return final_recommendations

# Function to compare recommendations across different similarity metrics
def compare_similarity_metrics(user_id, n_recommendations=5):
    # List of similarity metrics to compare
    similarity_metrics = ['cosine', 'euclidean', 'pearson']
    results = []

    # Generate recommendations for each metric
    for metric in similarity_metrics:
        user_based_recs = user_based_cf(user_id, n_recommendations, metric=metric)
        item_based_recs = item_based_cf(user_id, n_recommendations, metric=metric)
        
        # Model-based CF is independent of similarity metrics
        model_based_recs = model_based_cf(user_id, n_recommendations)
        
        # Ensemble may vary with metric
        ensemble_recs = ensemble_recommendations(user_id, n_recommendations)
        
        # Store results for each metric
        results.append({
            "Similarity Metric": metric,
            "User-Based Recommendations": user_based_recs,
            "Item-Based Recommendations": item_based_recs,
            "Model-Based Recommendations": model_based_recs,
            "Ensemble Recommendations": ensemble_recs,
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results to console
    print("\n--- Similarity Metric Comparison ---")
    print(results_df.to_string(index=False))
    return results_df

# Execute: Compare recommendations for a random user across different similarity metrics
user_id = ratings['User-ID'].sample(1, random_state=42).iloc[0]
print(f"User ID: {user_id}")
comparison_results = compare_similarity_metrics(user_id, n_recommendations=3)

print("\n\n")
