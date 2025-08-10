import pandas as pd

def test_data_loads():
    df = pd.read_csv('data/reviews.csv')
    assert not df.empty, "DataFrame is empty"
    assert 'Recommended IND' in df.columns, "Target column missing"
    assert df.isnull().sum().sum() == 0, "There are missing values"
    