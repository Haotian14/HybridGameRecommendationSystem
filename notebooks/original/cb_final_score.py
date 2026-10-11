import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# 加载数据
df = pd.read_csv('merged_filtered_games.csv')
game_df = df.drop_duplicates(subset='app_id').reset_index(drop=True)

# 拼接文本特征
def concat_features(row):
    return ' '.join([
        str(row.get('About the game', '')),
        str(row.get('Tags', '')),
        str(row.get('genre', '')),
        str(row.get('Categories', '')),
        str(row.get('Developers', '')),
        str(row.get('Publishers', ''))
    ])

game_df['combined_features'] = game_df.apply(concat_features, axis=1)

# TF-IDF 向量
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(game_df['About the game'].fillna(''))
app_id_to_index = {aid: i for i, aid in enumerate(game_df['app_id'])}

# BERT embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = model.encode(game_df['combined_features'].tolist(), show_progress_bar=True)

# genre one-hot
def split_and_strip(s):
    if pd.isna(s): return []
    return [x.strip() for x in str(s).replace(';', ',').split(',') if x.strip()]

game_df['genres_list'] = game_df['genre'].apply(split_and_strip)
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(game_df['genres_list'])

# price normalize
game_df['Price'] = pd.to_numeric(game_df['Price'], errors='coerce').fillna(0)
scaler = MinMaxScaler()
game_df['Price_norm'] = scaler.fit_transform(game_df[['Price']])


# ----------- 用户画像 -----------
def get_user_profile(user_id):
    played = df[(df['user_id'] == user_id)]
    if played.empty:
        return None

    valid_indices, weights = [], []
    for _, row in played.iterrows():
        app_id = row['app_id']
        if app_id in app_id_to_index:
            valid_indices.append(app_id_to_index[app_id])
            weights.append(row['hours'])

    if not valid_indices:
        return None

    weights = np.array(weights)
    tfidf_vector = np.asarray(tfidf_matrix[valid_indices].mean(axis=0)).flatten()
    content_vec = np.average(bert_embeddings[valid_indices], axis=0, weights=weights)
    genre_vec = np.average(genre_matrix[valid_indices], axis=0, weights=weights)
    price_pref = game_df.iloc[valid_indices]['Price_norm'].mean()

    return tfidf_vector, content_vec, genre_vec, price_pref


# ---------- 读取候选集并打分 ----------
candidates_df = pd.read_csv('top500_predictions.csv')
result_rows = []

for user_id in candidates_df['user_id'].unique():
    profile = get_user_profile(user_id)
    if profile is None:
        print(f"User {user_id} has no positive history.")
        continue

    tfidf_vector, user_content, user_genre, user_price = profile
    tfidf_vector_dense = np.asarray(tfidf_vector).reshape(1, -1)

    user_candidates = candidates_df[candidates_df['user_id'] == user_id]
    app_ids = user_candidates['app_id'].tolist()

    indices = [app_id_to_index[aid] for aid in app_ids if aid in app_id_to_index]
    if not indices:
        continue

    tfidf_scores = cosine_similarity(tfidf_vector_dense, tfidf_matrix[indices]).flatten()
    content_scores = np.dot(bert_embeddings[indices], user_content)
    genre_scores = genre_matrix[indices] @ user_genre
    price_scores = 1 - np.abs(game_df.iloc[indices]['Price_norm'].values - user_price)

    # 加权组合分数
    final_scores = (
        0.5 * tfidf_scores +
        0.45 * content_scores +
        0.15 * genre_scores +
        0.15 * price_scores
    )

    for idx, score in zip(indices, final_scores):
        result_rows.append({
            'user_id': user_id,
            'app_id': game_df.iloc[idx]['app_id'],
            'score': score
        })

# 输出为新CSV
output_df = pd.DataFrame(result_rows)
output_df.to_csv('cb_scored_recommendations.csv', index=False)
print("cb_scored_recommendations.csv")