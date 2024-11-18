import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 데이터 로드
users = pd.read_csv('data/userFavoriteGenre.csv')  # User별 장르 선호도
books = pd.read_csv('data/Filtered_Books.csv')  # ISBN과 Book-Title
genres = pd.read_csv('data/Genres.csv')  # ISBN별 장르 부합 점수

# ISBN을 문자열로 변환
books['ISBN'] = books['ISBN'].astype(str)
genres['ISBN'] = genres['ISBN'].astype(str)

# 장르 점수를 계산한 genres에서 장르별 점수 컬럼을 생성
# 예: 'Romance', 'Thriller' 등 각 책에 대해 장르의 부합 점수가 있음
genres_long = genres.melt(id_vars=['ISBN'], var_name='Genre', value_name='Genre-Similarity')

# 책 데이터와 장르 데이터 병합
books_with_genres = pd.merge(books, genres_long, on='ISBN', how='inner')

# 사용자별 장르 선호도 벡터 생성
def create_user_vector(user_row):
    # 각 장르별 점수 리스트 생성
    genres_list = ['Romance', 'Thriller', 'Fantasy', 'Science Fiction',
                   'Mystery', 'Non-Fiction', 'Historical', 'Horror',
                   'Adventure', 'Biography']
    return [user_row[genre] for genre in genres_list]

# 상위 n권 추천 함수
def recommend_top_books(user_id, users, books_with_genres, top_n=5):
    # 사용자 정보 확인
    user_row = users[users['User-ID'] == user_id]
    if user_row.empty:
        return f"User-ID {user_id}에 대한 정보가 없습니다."
    
    # 유저의 선호도 벡터 생성
    user_vector = create_user_vector(user_row.iloc[0])

    # 코사인 유사도 계산
    def calculate_similarity(book_genre_sim):
        # 유사도 계산 시 NaN 방지 (유효한 값이 있어야 계산 가능)
        if not np.any(np.isnan(book_genre_sim)):  # NaN값이 없을 경우에만 계산
            return cosine_similarity([user_vector], [book_genre_sim])[0][0]
        return 0  # NaN이 있으면 유사도 0으로 처리

    # Genre-Similarity를 벡터로 묶어 유사도 계산
    books_with_genres['Similarity'] = books_with_genres.groupby('ISBN')['Genre-Similarity'].transform(
        lambda x: calculate_similarity(x.values))

    # 추천할 책 중 중복되지 않도록 상위 n권 추천
    recommended_books = []
    unrated_books = books_with_genres.sort_values(by='Similarity', ascending=False)

    # 추천된 책 목록을 저장하고, 이미 추천된 책은 제외
    for index, row in unrated_books.iterrows():
        if len(recommended_books) >= top_n:
            break
        # 이미 추천된 책을 제외
        if row['ISBN'] not in [book['ISBN'] for book in recommended_books]:
            recommended_books.append({
                'ISBN': row['ISBN'],  # ISBN을 추가하여 중복 검사에 사용
                'Book-Title': row['Book-Title'],
                'Book-Author': row['Book-Author'],
                'Similarity': row['Similarity']
            })
    
    return recommended_books

# 예시: User-ID 1733에 대한 상위 5권 추천
user_id = 392
recommendations = recommend_top_books(user_id, users, books_with_genres, top_n=5)

# 추천 결과 출력
for recommendation in recommendations:
    print(f"Book Title: {recommendation['Book-Title']}")
    print(f"Book Author: {recommendation['Book-Author']}")
    print(f"Similarity: {recommendation['Similarity']}\n")
