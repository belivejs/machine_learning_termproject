import pandas as pd

# 추천 결과 로드
recommended_df = pd.read_csv('data/recommended_books.csv')

# 사용자 평점 데이터 로드
rating_data = pd.read_csv('data/Filtered_Ratings.csv')
book_data = pd.read_csv('data/Filtered_Books.csv')

# ISBN을 기준으로 데이터 병합
user_book_rating = pd.merge(rating_data, book_data, on='ISBN')

# 피벗 테이블 생성
book_user_rating = user_book_rating.pivot_table('Book-Rating', index='Book-Title', columns='User-ID')
book_user_rating.fillna(0, inplace=True)

# 추천된 책의 실제 평점 가져오기
actual_ratings = user_book_rating[user_book_rating['Book-Title'].isin(recommended_df['Book Title'])]

# 추천된 책과 실제 평점 비교
recommended_df['Predicted Rating'] = recommended_df['Similarity Score']

# MAE 계산
actual_ratings = actual_ratings[['Book-Title', 'Book-Rating']]
merged_df = recommended_df.merge(actual_ratings, left_on='Book Title', right_on='Book-Title', how='left')

mae = (merged_df['Book-Rating'] - merged_df['Predicted Rating']).abs().mean()
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# 모든 사용자에 대해 정확도 계산
user_ids = rating_data['User-ID'].unique()
accuracy_list = []

for user_id in user_ids:
    user_actual_books = user_book_rating[user_book_rating['User-ID'] == user_id]['Book-Title'].tolist()
    correct_predictions = len(set(recommended_df['Book Title']) & set(user_actual_books))
    accuracy = correct_predictions / len(recommended_df) if len(recommended_df) > 0 else 0
    accuracy_list.append(accuracy)

# 평균 정확도 계산
average_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
print(f'Average Accuracy: {average_accuracy:.10f}')
