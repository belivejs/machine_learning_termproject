import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

# 데이터 로딩 (원본과 같은 형태)
df_ratings = pd.read_csv('data/Filtered_Ratings.csv')
df_books = pd.read_csv('data/Filtered_Books.csv')

# 데이터 준비
df_user_book_ratings = df_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
matrix = df_user_book_ratings.to_numpy()
user_ratings_mean = np.mean(matrix, axis=1)
matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)

# Regularization 파라미터 설정
lambda_reg = 0.1  # Regularization 파라미터 (값을 조절하여 모델을 최적화)

# SVD에서 Regularization을 적용하여 행렬 분해 (Regularization 적용)
U, sigma, Vt = svds(matrix_user_mean, k=12)
sigma = np.diag(sigma)
U_reg = U
Vt_reg = Vt
sigma_reg = sigma + lambda_reg * np.eye(sigma.shape[0])  # Regularization 항 추가

# Regularized 예측값 계산
svd_user_predicted_ratings_reg = np.dot(np.dot(U_reg, sigma_reg), Vt_reg) + user_ratings_mean.reshape(-1, 1)
df_svd_preds_reg = pd.DataFrame(svd_user_predicted_ratings_reg, columns=df_user_book_ratings.columns)

# Regularization 없이 모델 학습 (Regularization 미사용)
U, sigma, Vt = svds(matrix_user_mean, k=12)
sigma = np.diag(sigma)
svd_user_predicted_ratings_no_reg = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
df_svd_preds_no_reg = pd.DataFrame(svd_user_predicted_ratings_no_reg, columns=df_user_book_ratings.columns)

# 예측 값과 실제 값 비교 (실제 값을 기준으로 RMSE 계산)
def compute_rmse(predicted_ratings, actual_ratings):
    return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

# 실제 평점 데이터와 예측된 평점 데이터를 비교
user_id = 6323
user_row_number = df_user_book_ratings.index.get_loc(user_id)
actual_ratings = matrix[user_row_number]  # 실제 평점
predicted_ratings_reg = svd_user_predicted_ratings_reg[user_row_number]  # Regularized 예측 평점
predicted_ratings_no_reg = svd_user_predicted_ratings_no_reg[user_row_number]  # Regularized 사용 안함

# RMSE 계산
rmse_reg = compute_rmse(predicted_ratings_reg, actual_ratings)
rmse_no_reg = compute_rmse(predicted_ratings_no_reg, actual_ratings)

# 결과 출력
print(f"Regularization 사용 시 RMSE: {rmse_reg}")
print(f"Regularization 사용 안 함 시 RMSE: {rmse_no_reg}")

# 추천 함수
def recommend_books(df_svd_preds, user_id, ori_book_df, ori_ratings_df, num_recommendations=5):
    if user_id not in df_user_book_ratings.index:
        raise ValueError(f"User ID {user_id} does not exist in the dataset.")
    
    user_row_number = df_user_book_ratings.index.get_loc(user_id)
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
    
    user_data = ori_ratings_df[ori_ratings_df['User-ID'] == user_id]
    user_history = user_data.merge(ori_book_df, on='ISBN').sort_values(['Book-Rating'], ascending=False)
    
    recommendations = ori_book_df[~ori_book_df['ISBN'].isin(user_history['ISBN'])]
    recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), on='ISBN')
    recommendations = recommendations.rename(columns={user_row_number: 'Predictions'}).sort_values('Predictions', ascending=False)
    
    return user_history, recommendations

# 사용자 추천 실행 (Regularization 사용 모델)
already_rated_reg, predictions_reg = recommend_books(df_svd_preds_reg, user_id, df_books, df_ratings, 10)

# 사용자 추천 실행 (Regularization 미사용 모델)
already_rated_no_reg, predictions_no_reg = recommend_books(df_svd_preds_no_reg, user_id, df_books, df_ratings, 10)

# 결과 출력
print("Regularization 사용 시 추천 결과:")
print(predictions_reg.head(30))

print("\nRegularization 사용 안 함 시 추천 결과:")
print(predictions_no_reg.head(30))
