import pandas as pd
from sklearn.decomposition import PCA
from typing import Literal


class DimensionalityReducer:
    """Class to model dimensionality reduction using PCA for the given dataset."""

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        self.feature_columns = data.columns
        self.model = None

    def reduce_pca(self) -> pd.DataFrame:
        """Reduces dimensions using Principal Component Analysis (PCA)."""
        pca = PCA(n_components=self.n_components)
        reduced_data = pca.fit_transform(self.data)
        self.model = pca  
        return pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(self.n_components)])

    def reduce(self, method: Literal['pca'] = 'pca') -> pd.DataFrame:
        """General method to apply dimensionality reduction using PCA."""
        if method == 'pca':
            return self.reduce_pca()
        else:
            raise ValueError("Unsupported reduction method. Currently, only 'pca' is available.")

    def inverse_transform(self, reduced_data: pd.DataFrame) -> pd.DataFrame:
        """Reverses the dimensionality reduction using the trained PCA model."""
        if isinstance(self.model, PCA):
            # PCA supports inverse transform
            original_data = self.model.inverse_transform(reduced_data)
            return pd.DataFrame(original_data, columns=self.feature_columns)
        else:
            raise NotImplementedError("Inverse transformation is only supported for PCA.")

