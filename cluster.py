import pandas as pd
import numpy as np
from etl.config import load_settings
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

#Load the dataset from MySQL database using SQLAlchemy
try:
    from sqlalchemy import create_engine
except ImportError as exc:
    raise RuntimeError(
        "MySQL loading requires SQLAlchemy and a MySQL driver. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc  

settings = load_settings()
engine = create_engine(settings.mysql_url)

query = """
SELECT
    `rank`,
    `last_rank`,
    `peak_rank`,
    `weeks`,
    `is_new`,
    `youtube_views`,
    `youtube_likes` ,
    `youtube_comments`,
    `genius_pageviews`,
    `genius_annotation_count`
FROM test_schema.song_week_stats;
"""

df = pd.read_sql(query, engine)

#feature engineering: create new feature yt like to view ratio and yt like to comment ratio
df['yt_like_to_view_ratio'] = df['youtube_likes'] / df['youtube_views']
df['yt_like_to_comment_ratio'] = df['youtube_likes'] / df['youtube_comments']  


#select features 
count_featueres = [
    "youtube_views",
    "youtube_likes",
    "youtube_comments",
    "genius_pageviews",
    "genius_annotation_count"
]

for col in count_featueres:
    df[col] = np.log1p(df[col])

numeric_features = [
    'rank',
    'last_rank',
    'peak_rank',
    'weeks',
    'youtube_views',
    'youtube_likes' ,
    'youtube_comments',
    'genius_pageviews',
    'genius_annotation_count',
    'yt_like_to_view_ratio',
    'yt_like_to_comment_ratio'
]   

categorical_features = [
    'is_new'
]

X = df[numeric_features + categorical_features]

#preprocessing steps (scaling, encoding, handling data skewness replacing inf values and filling missing values)

X = X.replace([np.inf, -np.inf], np.nan)

numeric_pipeline = Pipeline([
    (
        "imputer", SimpleImputer(strategy="median", add_indicator=True)
    ),(
        "scaler", StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)
    )
])

preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


X_processed = preprocessor.fit_transform(X)

#metrics for evaluating the model (k means)
kmeans_results = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_processed)
    silhouette_avg = silhouette_score(X_processed, labels)
    davies_bouldin_avg = davies_bouldin_score(X_processed, labels)
    calinski_harabasz_avg = calinski_harabasz_score(X_processed, labels)
    
    kmeans_results.append({
        "k": k,
        "silhouette_score": silhouette_avg,
        "davies_bouldin_score": davies_bouldin_avg,
        "calinski_harabasz_score": calinski_harabasz_avg
    })


kmeans_results = pd.DataFrame(kmeans_results)

print("\nKMeans Evaluation:")
print(kmeans_results)

#select k 
best_k = kmeans_results.loc[
    kmeans_results["silhouette_score"].idxmax(),
    "k"
]

best_k = int(best_k)

print(f"\nSelected K: {best_k}")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

kmeans_labels = kmeans.fit_predict(X_processed)

df["kmeans_cluster"] = kmeans_labels

#inspect results 
print("\nKMeans Cluster Counts:")
print(df["kmeans_cluster"].value_counts().sort_index())