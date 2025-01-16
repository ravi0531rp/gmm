from pathlib import Path
from typing import List
from urllib.request import urlretrieve
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class to load and preprocess the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = ['eu_econ_require', 'eu_political_require', 'eu_googov_require']   
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate rows from the DataFrame."""
        return df.drop_duplicates()

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Removes specified non-feature columns and sets index columns."""
        df = df.drop(columns=non_features, errors='ignore')
        return df.set_index(index)

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills NaN values with the median of each numeric column."""
        df = df.fillna(df.median())
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales only the numeric features using StandardScaler."""
        scaler = StandardScaler()
        df[:] = scaler.fit_transform(df)
        return df

    def preprocess_data(self) -> pd.DataFrame:
        """Combines all preprocessing steps and returns the processed DataFrame."""
        df = self.party_data.copy()
        print("Data Loaded")
        df = self.remove_duplicates(df)
        print("Duplicates Removed")
        df = self.remove_nonfeature_cols(df, self.non_features, self.index)
        print("Non Feature columns removed")
        df = self.handle_NaN_values(df)
        print("Nan handled")
        df = self.scale_features(df)
        print("Features scaled")
        return df

# Example usage
if __name__ == "__main__":
    data_loader = DataLoader()
    preprocessed_data = data_loader.preprocess_data()
