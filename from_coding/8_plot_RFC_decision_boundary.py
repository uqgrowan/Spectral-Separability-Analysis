from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns

#Plot boundaries for given components (reruns pipeline every time)
def plot_average_boundary(independent, dependent, n_comps=3, plot_components=(0, 1), depth= None, bootstrapping= True, sample_split =2, runs=100, 

                         preprocessing="standard scaler", reducer="pca", 
                         classifier="random forest classifier", test_split=0.3):
    """Plot the average decision boundary over multiple runs of the classification pipeline.
    
    Args:
        independent: Dataframe of spectral data
        dependent: 1 column Dataframe with class labels
        n_comps: total number of components to use for classification
        plot_components: tuple of (first_component, second_component) to plot (zero-based indexing)
        runs: number of runs to average over
        preprocessing, reducer, classifier: pipeline components to use
        test_split: proportion of data to use for testing
    """
    # Validate component selection
    if max(plot_components) >= n_comps:
        raise ValueError(f"Cannot plot component {max(plot_components)} when only calculating {n_comps} components")
    
    # Encode categorical labels
    le = LabelEncoder()
    dependent_encoded = le.fit_transform(dependent)
    class_names = le.classes_
    
    # Store all pipelines to average their predictions
    all_pipelines = []
    
    # Get the full range of the reduced data to set plot boundaries
    functions = {
        "random forest classifier": RandomForestClassifier(class_weight="balanced_subsample", min_samples_split= sample_split, min_samples_leaf= 1, max_features = None, bootstrap= bootstrapping, max_depth= depth, verbose = False),
        "pca": PCA(n_components=n_comps),
        "standard scaler": StandardScaler(),
    }
    
    # First run to get data bounds
    pipe = Pipeline([
        ("scaler", functions[preprocessing]),
        ("reducer", functions[reducer]),
        ("classifier", functions[classifier]),
    ])
    
    # Fit initial pipeline to get data bounds
    pipe.fit(independent, dependent_encoded)
    reduced_data = pipe[:-1].transform(independent)
    
    # Extract the components to plot
    plot_data = reduced_data[:, [plot_components[0], plot_components[1]]]
    
    # Calculate plot boundaries with some padding
    x_min, x_max = plot_data[:, 0].min() - 1, plot_data[:, 0].max() + 1
    y_min, y_max = plot_data[:, 1].min() - 1, plot_data[:, 1].max() + 1
    
    # Perform multiple runs
    for _ in range(runs):
        x_train, x_test, y_train, y_test = train_test_split(
            independent, dependent_encoded, test_size=test_split, stratify=dependent_encoded
        )
        
        pipe = Pipeline([
            ("scaler", functions[preprocessing]),
            ("reducer", functions[reducer]),
            ("classifier", functions[classifier]),
        ])
        
        pipe.fit(x_train, y_train)
        all_pipelines.append(pipe)
    
    # Create a custom predict function that averages all pipeline predictions
    def averaged_predict(X):
        # For the mesh grid points, we need to create fake values for the non-plotted components
        if X.shape[1] == 2:
            # Create array of zeros for non-plotted components
            full_X = np.zeros((X.shape[0], n_comps))
            full_X[:, plot_components[0]] = X[:, 0]
            full_X[:, plot_components[1]] = X[:, 1]
            X = pipe["reducer"].inverse_transform(full_X)
            
        predictions = np.array([pipe.predict(X) for pipe in all_pipelines])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
       
    # Create and plot the decision boundary
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Transform mesh points and get predictions
    mesh_predictions = averaged_predict(mesh_points)
    ax.contourf(xx, yy, mesh_predictions.reshape(xx.shape),
                alpha=0.3, cmap='viridis')
    
    # Plot the transformed data points
    scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1], 
                        c=dependent_encoded, cmap='viridis',
                        alpha=0.8)
    
    # Add colorbar with class names
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks(np.arange(len(class_names)))
    cbar.set_ticklabels(class_names)
    
    # Get explained variance ratios for the plotted components
    if reducer == "pca":
        var_ratio = pipe["reducer"].explained_variance_ratio_
        comp1_var = var_ratio[plot_components[0]] * 100
        comp2_var = var_ratio[plot_components[1]] * 100
        xlabel = f"Component {plot_components[0]+1} ({comp1_var:.1f}% explained var.)"
        ylabel = f"Component {plot_components[1]+1} ({comp2_var:.1f}% explained var.)"
    else:
        xlabel = f"Component {plot_components[0]+1}"
        ylabel = f"Component {plot_components[1]+1}"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Average Decision Boundary over {runs} runs\n'
                 f'Using {preprocessing}, {reducer}, and {classifier}')
    
    plt.tight_layout()
    return fig


#Run pipeline once and plot components from that
def run_pca_analysis(independent, dependent, n_comps=3, runs=100, 
                    preprocessing="standard scaler", reducer="pca", 
                    classifier="random forest classifier", test_split=0.3):
    """Run multiple iterations of the classification pipeline and store results.
    
    Args:
        independent: Dataframe of spectral data
        dependent: 1 column Dataframe with class labels
        n_comps: number of components to calculate
        runs: number of runs to average over
        preprocessing, reducer, classifier: pipeline components to use
        test_split: proportion of data to use for testing
    
    Returns:
        dict: Contains all the averaged results and data needed for plotting
    """
    # Encode categorical labels
    le = LabelEncoder()
    dependent_encoded = le.fit_transform(dependent)
    class_names = le.classes_
    
    functions = {
        "logistic regression": LogisticRegression(),
        "random forest classifier": RandomForestClassifier(class_weight="balanced_subsample"),
        "pca": PCA(n_components=n_comps),
        "kpca": KernelPCA(n_components=n_comps),
        "standard scaler": StandardScaler(),
        "svc": SVC()
    }
    
    # Store results across all runs
    all_reduced_data = []
    all_accuracies = []
    all_predictions_mesh = []
    
    # Create base pipeline for initial transform
    base_pipe = Pipeline([
        ("scaler", functions[preprocessing]),
        ("reducer", functions[reducer])
    ])
    
    # Transform all data once to get bounds for mesh grid
    base_pipe.fit(independent)
    full_reduced_data = base_pipe.transform(independent)
    
    # Create mesh grid that covers all components
    bounds = []
    for i in range(n_comps):
        min_val = full_reduced_data[:, i].min() - 1
        max_val = full_reduced_data[:, i].max() + 1
        bounds.append((min_val, max_val))
    
    # Create mesh grids for all possible component combinations
    mesh_grids = {}
    for i in range(n_comps):
        for j in range(i+1, n_comps):
            xx, yy = np.meshgrid(
                np.linspace(bounds[i][0], bounds[i][1], 100),
                np.linspace(bounds[j][0], bounds[j][1], 100)
            )
            mesh_grids[(i, j)] = (xx, yy)
    
    # Perform multiple runs
    for _ in range(runs):
        x_train, x_test, y_train, y_test = train_test_split(
            independent, dependent_encoded, test_size=test_split, stratify=dependent_encoded
        )
        
        pipe = Pipeline([
            ("scaler", functions[preprocessing]),
            ("reducer", functions[reducer]),
            ("classifier", functions[classifier]),
        ])
        
        pipe.fit(x_train, y_train)
        
        # Store accuracy
        accuracy = pipe.score(x_test, y_test)
        all_accuracies.append(accuracy)
        
        # Store reduced data
        reduced_data = pipe[:-1].transform(independent)
        all_reduced_data.append(reduced_data)
        
        # Get predictions for all mesh grids
        run_predictions = {}
        for comp_pair, (xx, yy) in mesh_grids.items():
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            # Create full-dimensional mesh points
            full_mesh = np.zeros((mesh_points.shape[0], n_comps))
            full_mesh[:, comp_pair[0]] = mesh_points[:, 0]
            full_mesh[:, comp_pair[1]] = mesh_points[:, 1]
            
            # Transform back to original space and get predictions
            mesh_original = pipe["reducer"].inverse_transform(full_mesh)
            predictions = pipe.predict(mesh_original)
            run_predictions[comp_pair] = predictions
        
        all_predictions_mesh.append(run_predictions)
    
    # Calculate averages and compile results
    results = {
        'reduced_data': np.mean(all_reduced_data, axis=0),
        'accuracy_mean': np.mean(all_accuracies),
        'accuracy_std': np.std(all_accuracies),
        'mesh_grids': mesh_grids,
        'class_names': class_names,
        'dependent_encoded': dependent_encoded,
        'n_comps': n_comps,
        'var_ratio': pipe["reducer"].explained_variance_ratio_ if reducer == "pca" else None,
        'pipeline_params': {
            'preprocessing': preprocessing,
            'reducer': reducer,
            'classifier': classifier,
            'runs': runs
        }
    }
    
    # Average mesh predictions
    avg_predictions = {}
    for comp_pair in mesh_grids.keys():
        all_preds = np.array([run[comp_pair] for run in all_predictions_mesh])
        avg_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)
        avg_predictions[comp_pair] = avg_pred
    
    results['mesh_predictions'] = avg_predictions
    
    return results

def plot_pca_components(results, comp1, comp2):
    """Create a decision boundary plot for specified components using stored results.
    
    Args:
        results: dict of results from run_pca_analysis
        comp1: first component to plot (zero-based indexing)
        comp2: second component to plot (zero-based indexing)
    
    Returns:
        matplotlib figure
    """
    if max(comp1, comp2) >= results['n_comps']:
        raise ValueError(f"Cannot plot component {max(comp1, comp2)} when only {results['n_comps']} components were calculated")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get the relevant reduced data for plotting
    plot_data = results['reduced_data'][:, [comp1, comp2]]
    
    # Plot the transformed data points
    scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1],
                        c=results['dependent_encoded'], cmap='viridis',
                        alpha=0.5)
    
    # Add colorbar with class names
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks(np.arange(len(results['class_names'])))
    cbar.set_ticklabels(results['class_names'])
    
    # Plot decision boundary
    xx, yy = results['mesh_grids'][(comp1, comp2)]
    mesh_predictions = results['mesh_predictions'][(comp1, comp2)]
    ax.contourf(xx, yy, mesh_predictions.reshape(xx.shape),
                alpha=0.3, cmap='viridis')
    
    # Set labels with variance ratios if available
    if results['var_ratio'] is not None:
        comp1_var = results['var_ratio'][comp1] * 100
        comp2_var = results['var_ratio'][comp2] * 100
        xlabel = f"Component {comp1+1} ({comp1_var:.1f}% explained var.)"
        ylabel = f"Component {comp2+1} ({comp2_var:.1f}% explained var.)"
    else:
        xlabel = f"Component {comp1+1}"
        ylabel = f"Component {comp2+1}"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    params = results['pipeline_params']
    ax.set_title(f'Average Decision Boundary over {params["runs"]} runs\n'
                 f'Using {params["preprocessing"]}, {params["reducer"]}, and {params["classifier"]}\n'
                 f'Mean Accuracy: {results["accuracy_mean"]:.3f} ± {results["accuracy_std"]:.3f}')
    
    plt.tight_layout()
    return fig




dep = pd.read_csv(r"Combined_analysis\New_reflectance_labels.csv").iloc[:,5]
indep =  pd.read_csv(r"Combined_analysis\Resampled\Landsat_9.csv").iloc[:,3:]#.drop([str(num) for num in range(753, 769)], axis = 1)


fig1 = plot_average_boundary(indep, dep, n_comps=6, plot_components=(3, 4), runs=100)
plt.show()


results = run_pca_analysis(indep, dep, n_comps=6, runs=10)
fig1 = plot_pca_components(results, 0, 1)
plt.show()
fig2 = plot_pca_components(results, 0, 2)
plt.show()
fig3 = plot_pca_components(results, 1, 2)
plt.show()

fig, axs = plt.subplots(2,2)
axs[0,0] = plot_pca_components(results, 0, 1)
axs[0,1] = plot_pca_components(results, 0, 2)
axs[0,0] = plot_pca_components(results, 0, 3)
axs[0,1] = plot_pca_components(results, 1, 2)
axs[0,0] = plot_pca_components(results, 1, 2)
axs[0,1] = plot_pca_components(results, 2, 3)
plt.show()
