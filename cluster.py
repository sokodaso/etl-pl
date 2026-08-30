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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt 
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

pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_processed)

#metrics for evaluating the model (k means)
kmeans_results = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, labels)
    davies_bouldin_avg = davies_bouldin_score(X_pca, labels)
    calinski_harabasz_avg = calinski_harabasz_score(X_pca, labels)
    
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

kmeans_labels = kmeans.fit_predict(X_pca)

df["kmeans_cluster"] = kmeans_labels

#inspect k means results 
print("\nKMeans Cluster Counts:")
print(df["kmeans_cluster"].value_counts().sort_index())


#dbscan 
dbscan_results = []

for eps in np.arange(0.5,2.5,0.05):
    dbscan = DBSCAN(
        eps=eps,
        min_samples=5
    )

    labels = dbscan.fit_predict(X_pca)

    #Number of actual clusters excluding noise
    unique_labels = set(labels)

    n_clusters = len( unique_labels - {-1})

    n_noise = list(labels).count(-1)

    #Ignore noise when calculating clustering metrics
    mask = labels != -1

    #Silhouette requires atleast 2 clusters
    if n_clusters >= 2 and mask.sum() > n_clusters:

        silhouette = silhouette_score(
            X_pca[mask],
            labels[mask]
        )

        davies_bouldin = davies_bouldin_score(
            X_pca[mask],
            labels[mask]
        )

        calinski_harabasz = calinski_harabasz_score(
            X_pca[mask],
            labels[mask]
        )
    else:
        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan

    dbscan_results.append({
        "eps": eps,
        "min_samples": 5,
        "clusters": n_clusters,
        "noise_points" : n_noise,
        "silhouette": silhouette,
        "davies_bouldin" : davies_bouldin,
        "calinski_harabasz": calinski_harabasz
    })

dbscan_results = pd.DataFrame(dbscan_results)

print("\nDBSCAN Evaluation:")
print(dbscan_results)

#select dbscan parameters
valid_dbscan = dbscan_results.dropna(
    subset=["silhouette"]
)

if not valid_dbscan.empty:
    best_dbscan = valid_dbscan.loc[
        valid_dbscan["silhouette"].idxmax()
    ]

    best_eps = best_dbscan["eps"]
    best_min_samples = int(
        best_dbscan["min_samples"]
    )

    print(
         f"\nBest DBSCAN parameters: "
        f"eps={best_eps}, "
        f"min_samples={best_min_samples}"
    )


#final dbscan 
dbscan = DBSCAN(
    eps = best_eps,
    min_samples= best_min_samples
)


dbscan_labels = dbscan.fit_predict(
    X_pca
)

df["dbscan_cluster"] = dbscan_labels


#inspect final db results 
if "dbscan_cluster" in df.columns:

    print("\nDBSCAN Cluster Counts:")
    print(
        df["dbscan_cluster"]
        .value_counts()
        .sort_index()
    )


'''
min_samples = 5

# Find the nearest neighbors
neighbors = NearestNeighbors(
    n_neighbors=min_samples
)

neighbors.fit(X_processed)

distances, indices = neighbors.kneighbors(X_processed)

# Distance to the kth nearest neighbor for every point
k_distances = distances[:, -1]

# Sort distances from smallest to largest
k_distances = np.sort(k_distances)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(k_distances)

plt.xlabel("Data points sorted by distance")
plt.ylabel(f"Distance to {min_samples}th nearest neighbor")
plt.title("K-Distance Graph for DBSCAN")

plt.grid()
plt.show()
'''