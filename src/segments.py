"""segments.py -- Part C tasks 1-2: K-means subscriber segmentation.
Run:  python src/segments.py
Reads data/clean/subscribers.csv (Part A). Writes outputs/segments.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from constants import OUTPUT_DIR, RANDOM_STATE, SEGMENTS_PATH, SUBSCRIBERS_CLEAN_PATH

CLUSTER_FEATURES = ["tenure_months", "hours_watched_per_week", "monthly_recharge_inr"]
CHOSEN_K = 4
NEW_SUBSCRIBER = {"tenure_months": 4, "hours_watched_per_week": 15, "monthly_recharge_inr": 520}


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def fit_kmeans_sweep(X_scaled):
    '''Fit K-means for a range of cluster numbers and print inertia and silhouette scores.

    Args:
        X_scaled (np.ndarray): The scaled feature matrix.
    '''
    section("Task 1: K-means sweep (K=2..8)")
    print(f"{'K':>3} {'inertia':>10} {'silhouette':>11}")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"{k:>3} {km.inertia_:>10.1f} {sil:>11.3f}")
    print("\nInertia always falls as K grows because adding another centroid can only reduce (never increase) "
          "the total within-cluster distance, so K=8 beats K=2 on inertia by construction. K=2's silhouette is "
          "the best here because splitting the data into just two big, well-separated blobs is the easiest "
          "shape to score well on average, not because two segments are the most useful split for the "
          "business. Neither metric alone picks K: inertia always prefers more clusters and silhouette always "
          "prefers fewer, so the final choice needs the elbow (where inertia stops dropping quickly) plus "
          "whether the resulting groups are distinct and actionable enough to act on.")


def profile_segments(df, labels, chosen_k):
    '''Profile the segments by computing summary statistics for each segment.

    Args:
        df (pd.DataFrame): The subscriber dataframe.
        labels (np.ndarray): The segment labels for each subscriber.
        chosen_k (int): The number of segments chosen.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The dataframe with segment labels added.
            - profile (pd.DataFrame): The profile dataframe with segment statistics.
            - premium_id (int): The segment id of the premium segment.
            - at_risk_id (int): The segment id of the at-risk segment.
    '''
    section(f"Task 2: profile segments (K={chosen_k})")
    df = df.copy()
    df["segment_id"] = labels
    # Add the segment labels to the dataframe for further analysis and profiling
    profile = df.groupby("segment_id").agg(
        size=("subscriber_id", "count"),
        mean_tenure_months=("tenure_months", "mean"),
        mean_hours_watched_per_week=("hours_watched_per_week", "mean"),
        mean_monthly_recharge_inr=("monthly_recharge_inr", "mean"),
        churn_rate=("churned", "mean"),
    ).round(2)
    print(profile.to_string())

    # Identify the premium and at-risk segments based on the profile statistics
    premium_id = profile["mean_monthly_recharge_inr"].idxmax()
    at_risk_id = profile["churn_rate"].idxmax()
    print(f"\nPremium segment (highest mean monthly recharge): segment_id={premium_id}")
    print(f"At-Risk segment (highest churn rate): segment_id={at_risk_id}")
    print("Segment ids/numbers are arbitrary labels with no inherent order; only the profile above determines "
          "the business name.")
    return df, profile, premium_id, at_risk_id


def name_segments(profile, premium_id, at_risk_id):
    '''Assign human-readable names to the segments based on their profile.

    Args:
        profile (pd.DataFrame): The profile dataframe with segment statistics.
        premium_id (int): The segment id of the premium segment.
        at_risk_id (int): The segment id of the at-risk segment.

    Returns:
        dict: A mapping from segment id to segment name.
    '''
    names = {premium_id: "Premium", at_risk_id: "At-Risk"}
    remaining = [i for i in profile.index if i not in names]
    remaining_sorted = sorted(remaining, key=lambda i: profile.loc[i, "mean_tenure_months"], reverse=True)
    extra_names = ["Loyal Core", "Casual Viewer", "Budget Starter"]
    for i, seg_id in enumerate(remaining_sorted):
        names[seg_id] = extra_names[i] if i < len(extra_names) else f"Segment {seg_id}"
    return names


def classify_new_subscriber(scaler, kmeans, names):
    '''Classify a new subscriber by computing distances to cluster centroids and comparing with K-means prediction.

    Args:
        scaler (StandardScaler): The fitted scaler used for the clustering features.
        kmeans (KMeans): The fitted KMeans model.
        names (dict): Mapping from segment id to segment name.

    Returns:
        int: The segment id of the nearest centroid.
    '''
    section("Task 2: hand-classify a new subscriber")
    new_point = pd.DataFrame([NEW_SUBSCRIBER])[CLUSTER_FEATURES]
    scaled = scaler.transform(new_point)[0]
    print(f"Scaled new-subscriber point: {scaled.round(3).tolist()}")

    distances = np.linalg.norm(kmeans.cluster_centers_ - scaled, axis=1)
    for seg_id, dist in enumerate(distances):
        print(f"  distance to centroid {seg_id} ({names[seg_id]}): {dist:.3f}")

    nearest = int(np.argmin(distances))
    predicted = int(kmeans.predict(scaled.reshape(1, -1))[0])
    print(f"Nearest centroid by hand: {nearest} ({names[nearest]}), distance {distances[nearest]:.3f}")
    print(f"kmeans.predict: {predicted} ({names[predicted]})")
    assert nearest == predicted, "hand computation disagrees with kmeans.predict"
    print("Hand computation agrees with kmeans.predict.")
    return nearest


def main():
    df = pd.read_csv(SUBSCRIBERS_CLEAN_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

    fit_kmeans_sweep(X_scaled)

    kmeans = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_scaled)

    profiled_df, profile, premium_id, at_risk_id = profile_segments(df, labels, CHOSEN_K)
    names = name_segments(profile, premium_id, at_risk_id)
    print("\nSegment names:")
    for seg_id, name in sorted(names.items()):
        print(f"  segment {seg_id}: {name}")
    profiled_df["segment_name"] = profiled_df["segment_id"].map(names)

    classify_new_subscriber(scaler, kmeans, names)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    profiled_df[["subscriber_id", "segment_id", "segment_name"]].to_csv(SEGMENTS_PATH, index=False)
    print(f"\nWrote {SEGMENTS_PATH} ({len(profiled_df)} rows)")


if __name__ == "__main__":
    main()
