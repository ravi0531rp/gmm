import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data using a Gaussian Mixture Model (GMM).
    1. Write a function to model the distribution of the political party dataset using GMM
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher-dimensional space
    as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names, n_components: int = 3):
        """
        Initialize the DensityEstimator with the provided data, dimensionality reducer, 
        feature names, and the number of components for GMM.
        """
        self.data = data
        # Access the trained PCA model from the dim_reducer instance
        self.dim_reducer_model = dim_reducer.model  # Accessing model from DimensionalityReducer
        self.feature_names = high_dim_feature_names
        self.gmm_model = None
        self.n_components = n_components  # Default to 3 components for GMM
        
    def fit_distribution(self) -> GaussianMixture:
        """Fit a Gaussian Mixture Model to the data."""
        gmm = GaussianMixture(n_components=self.n_components)
        gmm.fit(self.data)
        self.gmm_model = gmm
        return gmm

    def sample_from_distribution(self, n_samples: int = 10) -> np.ndarray:
        """Randomly sample 10 data points from the fitted GMM distribution."""
        if self.gmm_model is None:
            raise ValueError("The distribution is not fitted yet. Please call 'fit_distribution' first.")
        
        # Sample from the GMM
        sampled_data, _ = self.gmm_model.sample(n_samples)
        return sampled_data

    def map_back_to_original_space(self, reduced_data: pd.DataFrame) -> pd.DataFrame:
        """Map the sampled data back to the original high-dimensional space."""
        # Using inverse transform from PCA model
        return self.dim_reducer_model.inverse_transform(reduced_data)

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Predict the density of a given data point in the fitted GMM distribution."""
        if self.gmm_model is None:
            raise ValueError("The distribution is not fitted yet. Please call 'fit_distribution' first.")
        
        # Predict the log likelihood of the data points
        log_density = self.gmm_model.score_samples(new_data)  # Returns log of the density
        return np.exp(log_density)  # Return the actual density (exponentiate the log density)

