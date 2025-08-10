from starter.starter import feature_engineering
import pandas as pd

def test_pipeline_transform():
    df = pd.read_csv('data/reviews.csv')
    X = df.drop('Recommended IND', axis=1)
    transformed = feature_engineering.fit_transform(X)
    assert transformed.shape[0] == X.shape[0], "Row count mismatch after transform"
    