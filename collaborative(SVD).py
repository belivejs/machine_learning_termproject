import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds  # 추가
from sklearn.metrics.pairwise import cosine_similarity
import operator


df_ratings = pd.read_csv('data/Filtered_Ratings.csv')
df_books = pd.read_csv('data/Filtered_Books.csv')

print(df_ratings.columns)

df_user_book_ratings = df_ratings.pivot(
    index = 'User-ID',
    columns = 'ISBN',
    values='Book-Rating'
).fillna(0)

# print(df_user_book_ratings.head(5))

matrix = df_user_book_ratings.to_numpy()

user_ratings_mean = np.mean(matrix, axis = 1)

matrix_user_mean = matrix - user_ratings_mean.reshape(-1,1)

# print(matrix)

# print(pd.DataFrame(matrix_user_mean, columns = df_user_book_ratings.columns).head(5))

#SVD(특이값 분해) 사용

U, sigma, Vt = svds(matrix_user_mean, k = 12)
# print(U.shape)
# print(sigma.shape)
# print(Vt.shape)


sigma = np.diag(sigma)
# print(sigma.shape)



#SVD 특이값 분해를 통해 
# Matrix Factorization(행렬분해)를 기반으로 데이터 변경
svd_user_predicted_ratings = np.dot(np.dot(U,sigma),Vt) 
+ user_ratings_mean.reshape(-1,1)

df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, 
columns = df_user_book_ratings.columns)

# print(df_svd_preds.head(8))



def recommend_books(df_svd_preds, user_id, ori_book_df, ori_ratings_df, num_recommendations=5):
    # 사용자 ID 유효성 검사
    if user_id not in df_user_book_ratings.index:
        raise ValueError(f"User ID {user_id} does not exist in the dataset.")
    
    # 정확한 행 번호 계산
    user_row_number = df_user_book_ratings.index.get_loc(user_id)
    
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
    
    user_data = ori_ratings_df[ori_ratings_df['User-ID'] == user_id]
    user_history = user_data.merge(ori_book_df, on='ISBN').sort_values(['Book-Rating'], ascending=False)
    
    recommendations = ori_book_df[~ori_book_df['ISBN'].isin(user_history['ISBN'])]
    recommendations = recommendations.merge(
        pd.DataFrame(sorted_user_predictions).reset_index(), 
        on='ISBN'
    )
    recommendations = recommendations.rename(columns={user_row_number: 'Predictions'}).sort_values('Predictions')
    
    return user_history, recommendations



# 사용자 ID가 데이터셋에 있어야 함
user_id = int('191883')
already_rated, predictions = recommend_books(df_svd_preds, user_id, df_books, df_ratings, 10)

print("사용자가 평가한 책:")
print(already_rated.head(10))

print("추천 결과:")
print(predictions.head(10))




# already_rated, predictions = recommend_books(df_svd_preds, 20180, df_books, df_ratings, 10)


# print('마지막 결과')
# print(already_rated.head(10))


