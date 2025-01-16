from pathlib import Path
from matplotlib import pyplot
import pandas as pd
from loader import DataLoader
from viz import scatter_plot, plot_density_estimation_results, plot_finnish_parties
from dim_reducer import DimensionalityReducer
from estimator import DensityEstimator

if __name__ == "__main__":
    # Data pre-processing step
    ##### YOUR CODE GOES HERE #####
    data_loader = DataLoader()
    preprocessed_data = data_loader.preprocess_data()

    # Dimensionality reduction step
    ##### YOUR CODE GOES HERE #####
    reducer = DimensionalityReducer(preprocessed_data)
    reduced_dim_data = reducer.reduce(method='pca')

    # Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    ##### YOUR CODE GOES HERE #####

    density_estimator = DensityEstimator(reduced_dim_data, reducer, preprocessed_data.columns)
    _ = density_estimator.fit_distribution()

    # Plot density estimation results here
    ##### YOUR CODE GOES HERE #####
    plot_density_estimation_results(
        reduced_dim_data,
        density_estimator.gmm_model.predict(reduced_dim_data),
        density_estimator.gmm_model.means_,
        density_estimator.gmm_model.covariances_,
        title="Density Estimation Results"
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))


    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    ##### YOUR CODE GOES HERE #####
    scatter_plot(
        reduced_dim_data,
        color="b",
        splot=splot,
        label="Left/Right Wing Parties"
    )

    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")


     # Plot finnish parties here
    ##### YOUR CODE GOES HERE #####
    pca_result_df = pd.DataFrame(reduced_dim_data, columns=["PC1", "PC2"])
    pca_result_df.index = preprocessed_data.index
    plot_finnish_parties(pca_result_df)



    print("Analysis Complete")