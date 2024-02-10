import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Importing models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
from sklearn.manifold import LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE

@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def main():
    st.title("Hyper Parameter Tuning ")

    uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        st.sidebar.title('Model Selection')

        # Select dependent variable
        dependent_var = st.sidebar.selectbox('Select Dependent Variable', data.columns)
        independent_vars = data.columns.tolist()
        independent_vars.remove(dependent_var)  # Remove dependent variable from independent variables list

        # Select independent variables
        selected_independent_vars = st.sidebar.multiselect('Select Independent Variables', independent_vars)

        if len(selected_independent_vars) > 0:
            # Plot scatter plot if independent variables are selected
            st.write("### Scatter Plot of the selected variables")
            fig, ax = plt.subplots()
            for var in selected_independent_vars:
                ax.scatter(data[var], data[dependent_var], label=var)
            ax.set_xlabel('Independent Variables')
            ax.set_ylabel('Dependent Variable')
            ax.legend()
            st.pyplot(fig)
        else:
            # If no variables selected, display scatter plot based on variable types
            st.write("### Scatter Plot based on variable types")
            fig, ax = plt.subplots()

            # Get variable types
            variable_types = data.dtypes
            categorical_vars = variable_types[variable_types == 'object'].index
            numerical_vars = variable_types[variable_types != 'object'].index

            # Plot scatter plots for different combinations
            for cat_var in categorical_vars:
                for num_var in numerical_vars:
                    ax.scatter(data[cat_var], data[num_var], label=f'{cat_var} vs {num_var}')
            ax.set_xlabel('Categorical Variables')
            ax.set_ylabel('Numerical Variables')
            ax.legend()
            st.pyplot(fig)

        X = data[selected_independent_vars]
        y = data[dependent_var]

        # Create a dropdown for selecting the algorithm
        selected_model = st.sidebar.selectbox('Select Model', ('Elastic Net', 'Random Forest', 
        'Quantile Regressor', 'Cross Decomposition', 'SVM', 'Nearest Neighbours', 'Decision Trees', 
        'Gradient Boosting', 'Manifold Learning', 'Clustering'))

        if selected_model:
            st.sidebar.subheader('Model Hyperparameters')
            if selected_model == 'Elastic Net':
                # Elastic Net hyperparameters
                alpha = st.sidebar.slider('Alpha', 0.0, 1.0, 1.0)
                l1_ratio = st.sidebar.slider('L1 Ratio', 0.0, 1.0, 0.5)
                fit_intercept = st.sidebar.checkbox('Fit Intercept', value=True)
                precompute = st.sidebar.checkbox('Precompute', value=False)
                max_iter = st.sidebar.number_input('Max Iterations', min_value=1, max_value=10000, value=1000)
                copy_X = st.sidebar.checkbox('Copy X', value=True)
                tol = st.sidebar.number_input('Tolerance', min_value=0.0001, max_value=1.0, value=0.0001)
                warm_start = st.sidebar.checkbox('Warm Start', value=False)
                positive = st.sidebar.checkbox('Positive', value=False)
                random_state = st.sidebar.number_input('Random State', value=None)
                selection = st.sidebar.radio('Selection', ['cyclic', 'random'], index=0)

                st.sidebar.subheader('Train Model')
                if st.sidebar.button('Train Elastic Net Model'):
                    # Train Elastic Net
                    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, precompute=precompute, max_iter=max_iter,
                                                   copy_X=copy_X, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection)
                    elastic_net_model.fit(X, y)
                    y_pred = elastic_net_model.predict(X)
                    accuracy = accuracy_score(y, y_pred)
                    st.write("Elastic Net Model trained.")
                    st.write("Accuracy:", accuracy)

            elif selected_model == 'Random Forest':
                st.sidebar.subheader('Random Forest Parameters')
                selected_model_type = st.sidebar.radio('Model Type', ['Classification', 'Regression'])
                if selected_model_type == 'Classification':
                    # Random Forest Classification hyperparameters
                    n_estimators = st.sidebar.number_input('Number of Estimators', min_value=1, max_value=1000, value=100)
                    criterion = st.sidebar.radio('Criterion', ['gini', 'entropy'], index=0)
                    max_depth = st.sidebar.number_input('Max Depth', min_value=1, max_value=100, value=None)
                    min_samples_split = st.sidebar.number_input('Min Samples Split', min_value=2, max_value=20, value=2)
                    min_samples_leaf = st.sidebar.number_input('Min Samples Leaf', min_value=1, max_value=20, value=1)
                    min_weight_fraction_leaf = st.sidebar.number_input('Min Weight Fraction Leaf', min_value=0.0, max_value=0.5, value=0.0)
                    max_features = st.sidebar.radio('Max Features', ['sqrt', 'log2'], index=0)
                    max_leaf_nodes = st.sidebar.number_input('Max Leaf Nodes', min_value=2, max_value=100, value=None)
                    min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0)
                    bootstrap = st.sidebar.checkbox('Bootstrap', value=True)
                    oob_score = st.sidebar.checkbox('Out-of-bag Score', value=False)
                    n_jobs = st.sidebar.number_input('Number of Jobs', min_value=-1, max_value=16, value=None)
                    random_state = st.sidebar.number_input('Random State', value=None)
                    verbose = st.sidebar.number_input('Verbose', value=0)
                    warm_start = st.sidebar.checkbox('Warm Start', value=False)
                    class_weight = st.sidebar.radio('Class Weight', ['None', 'balanced'], index=0)
                    ccp_alpha = st.sidebar.number_input('CCP Alpha', min_value=0.0, max_value=0.5, value=0.0)
                    max_samples = st.sidebar.number_input('Max Samples', min_value=1, max_value=len(X), value=None)
                    monotonic_cst = st.sidebar.checkbox('Monotonic Constraint', value=None)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Random Forest Classification Model'):
                        # Train Random Forest Classification
                        random_forest_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                                      min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                                      max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                                      bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs,
                                                                      random_state=random_state, verbose=verbose, warm_start=warm_start,
                                                                      class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples,
                                                                      monotonic_cst=monotonic_cst)
                        random_forest_model.fit(X, y)
                        y_pred_rf = random_forest_model.predict(X)
                        accuracy_rf = accuracy_score(y, y_pred_rf)
                        st.write("Random Forest Classification Model trained.")
                        st.write("Accuracy:", accuracy_rf)

                elif selected_model_type == 'Regression':
                    # Random Forest Regression hyperparameters
                    n_estimators = st.sidebar.number_input('Number of Estimators', min_value=1, max_value=1000, value=100)
                    criterion = st.sidebar.radio('Criterion', ['squared_error', 'mse'], index=0)
                    max_depth = st.sidebar.number_input('Max Depth', min_value=1, max_value=100, value=None)
                    min_samples_split = st.sidebar.number_input('Min Samples Split', min_value=2, max_value=20, value=2)
                    min_samples_leaf = st.sidebar.number_input('Min Samples Leaf', min_value=1, max_value=20, value=1)
                    min_weight_fraction_leaf = st.sidebar.number_input('Min Weight Fraction Leaf', min_value=0.0, max_value=0.5, value=0.0)
                    max_features = st.sidebar.number_input('Max Features', min_value=0.1, max_value=1.0, value=1.0, step=0.1)
                    max_leaf_nodes = st.sidebar.number_input('Max Leaf Nodes', min_value=2, max_value=100, value=None)
                    min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0)
                    bootstrap = st.sidebar.checkbox('Bootstrap', value=True)
                    oob_score = st.sidebar.checkbox('Out-of-bag Score', value=False)
                    n_jobs = st.sidebar.number_input('Number of Jobs', min_value=-1, max_value=16, value=None)
                    random_state = st.sidebar.number_input('Random State', value=None)
                    verbose = st.sidebar.number_input('Verbose', value=0)
                    warm_start = st.sidebar.checkbox('Warm Start', value=False)
                    ccp_alpha = st.sidebar.number_input('CCP Alpha', min_value=0.0, max_value=0.5, value=0.0)
                    max_samples = st.sidebar.number_input('Max Samples', min_value=1, max_value=len(X), value=None)
                    monotonic_cst = st.sidebar.checkbox('Monotonic Constraint', value=None)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Random Forest Regression Model'):
                        # Train Random Forest Regression
                        random_forest_model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                                     min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                                     max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                                     bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs,
                                                                     random_state=random_state, verbose=verbose, warm_start=warm_start,
                                                                     ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst)
                        random_forest_model.fit(X, y)
                        y_pred_rf = random_forest_model.predict(X)
                        mse = mean_squared_error(y, y_pred_rf)
                        st.write("Random Forest Regression Model trained.")
                        st.write("Mean Squared Error:", mse)

            elif selected_model == 'Quantile Regressor':
                # Quantile Regressor hyperparameters
                quantile = st.sidebar.slider('Quantile', 0.0, 1.0, 0.5)
                alpha = st.sidebar.slider('Alpha', 0.0, 1.0, 1.0)
                fit_intercept = st.sidebar.checkbox('Fit Intercept', value=True)
                solver = st.sidebar.radio('Solver', ['highs', 'default'], index=0)
                solver_options = None

                st.sidebar.subheader('Train Model')
                if st.sidebar.button('Train Quantile Regressor Model'):
                    # Train Quantile Regressor
                    quantile_regressor_model = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=fit_intercept, solver=solver, solver_options=solver_options)
                    quantile_regressor_model.fit(X, y)
                    quantiles_pred = quantile_regressor_model.predict(X)
                    quantile_loss = np.mean(np.maximum(quantile * (y - quantiles_pred), (quantile - 1) * (y - quantiles_pred)))
        
                    st.write("Quantile Regressor Model trained.")
                    st.write("Quantile Loss:", quantile_loss)


            elif selected_model == 'Cross Decomposition':
                # Cross Decomposition hyperparameters
                n_components = st.sidebar.number_input('Number of Components', min_value=2, max_value=10, value=2)
                scale = st.sidebar.checkbox('Scale', value=True)
                max_iter = st.sidebar.number_input('Max Iterations', min_value=100, max_value=1000, value=500)
                tol = st.sidebar.number_input('Tolerance', min_value=1e-6, max_value=1e-3, value=1e-6)
                copy = st.sidebar.checkbox('Copy', value=True)

                st.sidebar.subheader('Train Model')
                if st.sidebar.button('Train Cross Decomposition Model'):
                    # Train Cross Decomposition
                    cross_decomposition_model = PLSRegression(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol, copy=copy)
                    cross_decomposition_model.fit(X, y)
                    # Compute accuracy (as it's a regression model, accuracy might not be the right term here)
                    st.write("Cross Decomposition Model trained.")

            elif selected_model == 'SVM':
                # SVM hyperparameters
                penalty = st.sidebar.radio('Penalty', ['l1', 'l2'], index=1)
                loss = st.sidebar.radio('Loss', ['hinge', 'squared_hinge'], index=1)
                dual = st.sidebar.radio('Dual', ['warn', 'auto'], index=0)
                tol = st.sidebar.number_input('Tolerance', value=0.0001)
                C = st.sidebar.number_input('C', value=1.0)
                multi_class = st.sidebar.radio('Multi Class', ['ovr', 'crammer_singer'], index=0)
                fit_intercept = st.sidebar.checkbox('Fit Intercept', value=True)
                intercept_scaling = st.sidebar.number_input('Intercept Scaling', value=1)
                class_weight = st.sidebar.radio('Class Weight', ['None', 'balanced'], index=0)
                verbose = st.sidebar.number_input('Verbose', value=0)
                random_state = st.sidebar.number_input('Random State', value=None)
                max_iter = st.sidebar.number_input('Max Iterations', value=1000)

                st.sidebar.subheader('Train Model')
                if st.sidebar.button('Train SVM Model'):
                    # Train SVM
                    svm_model = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class,
                                          fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                          class_weight=class_weight, verbose=verbose, random_state=random_state,
                                          max_iter=max_iter)
                    svm_model.fit(X, y)
                    y_pred_svm = svm_model.predict(X)
                    accuracy_svm = accuracy_score(y, y_pred_svm)
                    st.write("SVM Model trained.")
                    st.write("Accuracy:", accuracy_svm)

            elif selected_model == 'Nearest Neighbours':
                # Nearest Neighbours hyperparameters
                n_neighbors = st.sidebar.number_input('Number of Neighbours', min_value=1, max_value=100, value=5)
                weights = st.sidebar.radio('Weights', ['uniform', 'distance'], index=0)
                algorithm = st.sidebar.radio('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
                leaf_size = st.sidebar.number_input('Leaf Size', min_value=1, max_value=50, value=30)
                p = st.sidebar.number_input('P', min_value=1, max_value=2, value=2)
                metric = st.sidebar.radio('Metric', ['minkowski', 'euclidean', 'manhattan'], index=0)
                n_jobs = st.sidebar.number_input('Number of Jobs', min_value=1, max_value=-1, value=None)

                st.sidebar.subheader('Train Model')
                if st.sidebar.button('Train Nearest Neighbours Model'):
                    # Train Nearest Neighbours
                    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                                     leaf_size=leaf_size, p=p, metric=metric, n_jobs=n_jobs, )
                    knn_model.fit(X, y)
                    y_pred_knn = knn_model.predict(X)
                    accuracy_knn = accuracy_score(y, y_pred_knn)
                    st.write("Nearest Neighbours Model trained.")
                    st.write("Accuracy:", accuracy_knn)

            elif selected_model == 'Decision Trees':
                # Decision Trees hyperparameters
                criterion = st.sidebar.radio('Criterion', ['gini', 'entropy'], index=0)
                splitter = st.sidebar.radio('Splitter', ['best', 'random'], index=0)
                max_depth = st.sidebar.number_input('Max Depth', min_value=1, max_value=100, value=None)
                min_samples_split = st.sidebar.number_input('Min Samples Split', min_value=2, max_value=20, value=2)
                min_samples_leaf = st.sidebar.number_input('Min Samples Leaf', min_value=1, max_value=20, value=1)
                min_weight_fraction_leaf = st.sidebar.number_input('Min Weight Fraction Leaf', min_value=0.0, max_value=0.5, value=0.0)
                max_features = st.sidebar.radio('Max Features', ['None', 'sqrt', 'log2'], index=0)
                max_leaf_nodes = st.sidebar.number_input('Max Leaf Nodes', min_value=2, max_value=100, value=None)
                min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0)
                ccp_alpha = st.sidebar.number_input('CCP Alpha', min_value=0.0, max_value=0.5, value=0.0)

                st.sidebar.subheader('Train Model')
                if st.sidebar.button('Train Decision Trees Model'):
                    # Train Decision Trees
                    decision_tree_model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                                 min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                                 max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                                 ccp_alpha=ccp_alpha)
                    decision_tree_model.fit(X, y)
                    y_pred_dt = decision_tree_model.predict(X)
                    accuracy_dt = accuracy_score(y, y_pred_dt)
                    st.write("Decision Trees Model trained.")
                    st.write("Accuracy:", accuracy_dt)

            elif selected_model == 'Gradient Boosting':
                # Gradient Boosting hyperparameters
                loss = st.sidebar.radio('Loss', ['deviance', 'exponential'], index=0)
                learning_rate = st.sidebar.number_input('Learning Rate', min_value=0.01, max_value=1.0, value=0.1)
                n_estimators = st.sidebar.number_input('Number of Estimators', min_value=1, max_value=1000, value=100)
                subsample = st.sidebar.slider('Subsample', 0.1, 1.0, 1.0)
                criterion_gb = st.sidebar.radio('Criterion', ['friedman_mse', 'mse', 'mae'], index=0)
                min_samples_split_gb = st.sidebar.number_input('Min Samples Split', min_value=2, max_value=20, value=2)
                min_samples_leaf_gb = st.sidebar.number_input('Min Samples Leaf', min_value=1, max_value=20, value=1)
                min_weight_fraction_leaf_gb = st.sidebar.number_input('Min Weight Fraction Leaf', min_value=0.0, max_value=0.5, value=0.0)
                max_depth_gb = st.sidebar.number_input('Max Depth', min_value=1, max_value=100, value=3)
                min_impurity_decrease_gb = st.sidebar.number_input('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0)
                init = st.sidebar.radio('Init', ['None', 'zero'], index=0)
                random_state_gb = st.sidebar.number_input('Random State', value=None)
                max_features_gb = st.sidebar.radio('Max Features', ['None', 'sqrt', 'log2'], index=0)
                verbose_gb = st.sidebar.number_input('Verbose', value=0)
                max_leaf_nodes_gb = st.sidebar.number_input('Max Leaf Nodes', min_value=2, max_value=100, value=None)
                warm_start_gb = st.sidebar.checkbox('Warm Start', value=False)
                validation_fraction = st.sidebar.number_input('Validation Fraction', min_value=0.01, max_value=0.5, value=0.1)
                n_iter_no_change = st.sidebar.number_input('Number of Iterations No Change', min_value=1, max_value=20, value=None)
                tol_gb = st.sidebar.number_input('Tolerance', min_value=1e-4, max_value=1e-2, value=1e-4)
                ccp_alpha_gb = st.sidebar.number_input('CCP Alpha', min_value=0.0, max_value=0.5, value=0.0)

                st.sidebar.subheader('Train Model')
                if st.sidebar.button('Train Gradient Boosting Model'):
                    # Train Gradient Boosting
                    gradient_boosting_model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                                                         subsample=subsample, criterion=criterion_gb, min_samples_split=min_samples_split_gb,
                                                                         min_samples_leaf=min_samples_leaf_gb, min_weight_fraction_leaf=min_weight_fraction_leaf_gb,
                                                                         max_depth=max_depth_gb, min_impurity_decrease=min_impurity_decrease_gb, init=init,
                                                                         random_state=random_state_gb, max_features=max_features_gb, verbose=verbose_gb,
                                                                         max_leaf_nodes=max_leaf_nodes_gb, warm_start=warm_start_gb,
                                                                         validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                                                                         tol=tol_gb, ccp_alpha=ccp_alpha_gb)
                    gradient_boosting_model.fit(X, y)
                    y_pred_gb = gradient_boosting_model.predict(X)
                    accuracy_gb = accuracy_score(y, y_pred_gb)
                    st.write("Gradient Boosting Model trained.")
                    st.write("Accuracy:", accuracy_gb)

            elif selected_model == 'Manifold Learning':
                st.sidebar.subheader('Select Manifold Learning Algorithm')
                manifold_algorithm = st.sidebar.selectbox('Algorithm', ['Isomap', 'Locally Linear Embedding','Spectral Embedding','t-SNE'])

                if manifold_algorithm == 'Isomap':
                    n_neighbors_iso = st.sidebar.number_input('Number of Neighbours', min_value=2, max_value=100, value=5)
                    n_components_iso = st.sidebar.number_input('Number of Components', min_value=2, max_value=10, value=2)
                    eigen_solver_iso = st.sidebar.selectbox('Eigen Solver', ['auto', 'arpack', 'dense'])
                    tol_iso = st.sidebar.number_input('Tolerance', min_value=0, max_value=1, value=0)
                    max_iter_iso = st.sidebar.number_input('Max Iterations', min_value=1, max_value=1000, value=None)
                    path_method_iso = st.sidebar.selectbox('Path Method', ['auto', 'FW', 'D', 'auto'])
                    neighbors_algorithm_iso = st.sidebar.selectbox('Neighbors Algorithm', ['auto', 'brute', 'kd_tree', 'ball_tree'])
                    n_jobs_iso = st.sidebar.number_input('Number of Jobs', min_value=1, max_value=-1, value=None)
                    metric_iso = st.sidebar.selectbox('Metric', ['minkowski', 'euclidean', 'manhattan'])
                    p_iso = st.sidebar.number_input('P', min_value=1, max_value=2, value=2)
                    metric_params_iso = None

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Isomap Model'):
                        iso_model = Isomap(n_neighbors=n_neighbors_iso, n_components=n_components_iso, eigen_solver=eigen_solver_iso,
                           tol=tol_iso, max_iter=max_iter_iso, path_method=path_method_iso,
                           neighbors_algorithm=neighbors_algorithm_iso, n_jobs=n_jobs_iso,
                           metric=metric_iso, p=p_iso, metric_params=metric_params_iso)
                        X_iso = iso_model.fit_transform(X, y)
                        # Visualize manifold learning results
                        fig_iso = plt.figure()
                        plt.scatter(X_iso[:, 0], X_iso[:, 1], c=y)
                        plt.title('Isomap Manifold Learning')
                        plt.xlabel('Component 1')
                        plt.ylabel('Component 2')
                        st.pyplot(fig_iso)

                elif manifold_algorithm == 'Locally Linear Embedding':
                    n_neighbors_lle = st.sidebar.number_input('Number of Neighbours', min_value=2, max_value=100, value=5)
                    n_components_lle = st.sidebar.number_input('Number of Components', min_value=2, max_value=10, value=2)
                    reg_lle = st.sidebar.number_input('Regularization', min_value=0, value=0.001)
                    eigen_solver_lle = st.sidebar.selectbox('Eigen Solver', ['auto', 'arpack', 'dense'])
                    tol_lle = st.sidebar.number_input('Tolerance', min_value=0, value=1e-06)
                    max_iter_lle = st.sidebar.number_input('Max Iterations', min_value=1, max_value=1000, value=100)
                    method_lle = st.sidebar.selectbox('Method', ['standard', 'hessian', 'modified'])
                    hessian_tol_lle = st.sidebar.number_input('Hessian Tolerance', min_value=0, value=0.0001)
                    modified_tol_lle = st.sidebar.number_input('Modified Tolerance', min_value=0, value=1e-12)
                    neighbors_algorithm_lle = st.sidebar.selectbox('Neighbors Algorithm', ['auto', 'brute', 'kd_tree', 'ball_tree'])
                    random_state_lle = st.sidebar.number_input('Random State', value=None)
                    n_jobs_lle = st.sidebar.number_input('Number of Jobs', min_value=1, max_value=-1, value=None)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Locally Linear Embedding Model'):
                        lle_model = LocallyLinearEmbedding(n_neighbors=n_neighbors_lle, n_components=n_components_lle, reg=reg_lle,
                                           eigen_solver=eigen_solver_lle, tol=tol_lle, max_iter=max_iter_lle,
                                           method=method_lle, hessian_tol=hessian_tol_lle,
                                           modified_tol=modified_tol_lle, neighbors_algorithm=neighbors_algorithm_lle,
                                           random_state=random_state_lle, n_jobs=n_jobs_lle)
                    X_lle = lle_model.fit_transform(X, y)
        # Visualize manifold learning results
                    fig_lle = plt.figure()
                    plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y)
                    plt.title('Locally Linear Embedding Manifold Learning')
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    st.pyplot(fig_lle)

                elif manifold_algorithm == 'Spectral Embedding':
                    n_components_se = st.sidebar.number_input('Number of Components', min_value=2, max_value=10, value=2)
                    affinity_se = st.sidebar.selectbox('Affinity', ['nearest_neighbors', 'rbf'])
                    gamma_se = st.sidebar.number_input('Gamma', value=None)
                    random_state_se = st.sidebar.number_input('Random State', value=None)
                    eigen_solver_se = st.sidebar.selectbox('Eigen Solver', [None, 'arpack', 'lobpcg', 'amg'])
                    eigen_tol_se = st.sidebar.selectbox('Eigen Tolerance', ['auto', 0.0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1])
                    n_neighbors_se = st.sidebar.number_input('Number of Neighbours', min_value=2, max_value=100, value=None)
                    n_jobs_se = st.sidebar.number_input('Number of Jobs', min_value=1, max_value=-1, value=None)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Spectral Embedding Model'):
                        se_model = SpectralEmbedding(n_components=n_components_se, affinity=affinity_se, gamma=gamma_se,
                                     random_state=random_state_se, eigen_solver=eigen_solver_se,
                                     eigen_tol=eigen_tol_se, n_neighbors=n_neighbors_se, n_jobs=n_jobs_se)
                        X_se = se_model.fit_transform(X, y)
        # Visualize manifold learning results
                        fig_se = plt.figure()
                        plt.scatter(X_se[:, 0], X_se[:, 1], c=y)
                        plt.title('Spectral Embedding Manifold Learning')
                        plt.xlabel('Component 1')
                        plt.ylabel('Component 2')
                        st.pyplot(fig_se)

                elif manifold_algorithm == 't-SNE':
                    n_components_tsne = st.sidebar.number_input('Number of Components', min_value=2, max_value=10, value=2)
                    perplexity_tsne = st.sidebar.number_input('Perplexity', min_value=5, value=30.0)
                    early_exaggeration_tsne = st.sidebar.number_input('Early Exaggeration', min_value=1.0, value=12.0)
                    learning_rate_tsne = st.sidebar.selectbox('Learning Rate', ['auto', 'constant', 'optimal', 'adaptive'], index=0)
                    n_iter_tsne = st.sidebar.number_input('Number of Iterations', min_value=250, value=1000)
                    n_iter_without_progress_tsne = st.sidebar.number_input('Number of Iterations Without Progress', min_value=50, value=300)
                    min_grad_norm_tsne = st.sidebar.number_input('Minimum Gradient Norm', value=1e-07)
                    metric_tsne = st.sidebar.selectbox('Metric', ['euclidean', 'manhattan', 'cosine'], index=0)
                    metric_params_tsne = None
                    init_tsne = st.sidebar.selectbox('Initialization', ['pca', 'random'], index=0)
                    verbose_tsne = st.sidebar.number_input('Verbose', min_value=0, value=0)
                    random_state_tsne = st.sidebar.number_input('Random State', value=None)
                    method_tsne = st.sidebar.selectbox('Method', ['barnes_hut', 'exact'], index=0)
                    angle_tsne = st.sidebar.number_input('Angle', min_value=0.1, max_value=0.8, value=0.5)
                    n_jobs_tsne = st.sidebar.number_input('Number of Jobs', min_value=1, max_value=-1, value=None)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train t-SNE Model'):
                        tsne_model = TSNE(n_components=n_components_tsne, perplexity=perplexity_tsne, early_exaggeration=early_exaggeration_tsne,
                          learning_rate=learning_rate_tsne, n_iter=n_iter_tsne, n_iter_without_progress=n_iter_without_progress_tsne,
                          min_grad_norm=min_grad_norm_tsne, metric=metric_tsne, metric_params=metric_params_tsne, init=init_tsne,
                          verbose=verbose_tsne, random_state=random_state_tsne, method=method_tsne, angle=angle_tsne, n_jobs=n_jobs_tsne)
                        X_tsne = tsne_model.fit_transform(X, y)
        # Visualize manifold learning results
                        fig_tsne = plt.figure()
                        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
                        plt.title('t-SNE Manifold Learning')
                        plt.xlabel('Component 1')
                        plt.ylabel('Component 2')
                        st.pyplot(fig_tsne)


            elif selected_model == 'Clustering' : 
                cluster_type  = st.sidebar.selectbox('Select Cluster Type', ['K-means', 'Affinity Propogation', 'Mean Shift', 'Spectral Clustering', 'Agglomerative', 'DBSCAN', 'HDBSCAN'])
                if cluster_type == 'K-means':
                    n_clusters_km = st.sidebar.number_input('Number of Clusters', min_value=2, max_value=20, value=8)
                    init_km = st.sidebar.selectbox('Initialization', ['k-means++', 'random'], index=0)
                    n_init_km = st.sidebar.number_input('Number of Initializations', min_value=1, value=10)
                    max_iter_km = st.sidebar.number_input('Max Iterations', min_value=1, value=300)
                    tol_km = st.sidebar.number_input('Tolerance', min_value=0.0, value=0.0001)
                    verbose_km = st.sidebar.number_input('Verbose', min_value=0, value=0)
                    random_state_km = st.sidebar.number_input('Random State', value=None)
                    copy_x_km = st.sidebar.checkbox('Copy X', value=True)
                    algorithm_km = st.sidebar.selectbox('Algorithm', ['auto', 'full', 'elkan'], index=0)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train K-Means Model'):
                        kmeans_model = KMeans(n_clusters=n_clusters_km, init=init_km, n_init=n_init_km, max_iter=max_iter_km, tol=tol_km,
                                  verbose=verbose_km, random_state=random_state_km, copy_x=copy_x_km, algorithm=algorithm_km)
                        labels_km = kmeans_model.fit_predict(X)
            # Visualize clustering results
                        fig_km = plt.figure()
                        plt.scatter(X[:, 0], X[:, 1], c=labels_km, cmap='viridis')
                        plt.title('K-Means Clustering')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        st.pyplot(fig_km)

                elif cluster_type == 'Affinity Propogation':
                    damping_ap = st.sidebar.slider('Damping', 0.0, 1.0, 0.5)
                    max_iter_ap = st.sidebar.number_input('Max Iterations', min_value=1, value=200)
                    convergence_iter_ap = st.sidebar.number_input('Convergence Iterations', min_value=1, value=15)
                    copy_ap = st.sidebar.checkbox('Copy', value=True)
                    preference_ap = st.sidebar.number_input('Preference', value=None)
                    affinity_ap = st.sidebar.selectbox('Affinity', ['euclidean', 'precomputed'], index=0)
                    verbose_ap = st.sidebar.checkbox('Verbose', value=False)
                    random_state_ap = st.sidebar.number_input('Random State', value=None)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Affinity Propagation Model'):
                        affinity_propagation_model = AffinityPropagation(damping=damping_ap, max_iter=max_iter_ap, convergence_iter=convergence_iter_ap,
                                                             copy=copy_ap, preference=preference_ap, affinity=affinity_ap, verbose=verbose_ap,
                                                             random_state=random_state_ap)
                        labels_ap = affinity_propagation_model.fit_predict(X)
            # Visualize clustering results
                        fig_ap = plt.figure()
                        plt.scatter(X[:, 0], X[:, 1], c=labels_ap, cmap='viridis')
                        plt.title('Affinity Propagation Clustering')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        st.pyplot(fig_ap)

                elif cluster_type == 'Mean shift ' :
                    bandwidth_ms = st.sidebar.number_input('Bandwidth', value=None)
                    seeds_ms = st.sidebar.text_input('Seeds', value=None)
                    bin_seeding_ms = st.sidebar.checkbox('Bin Seeding', value=False)
                    min_bin_freq_ms = st.sidebar.number_input('Min Bin Frequency', min_value=1, value=1)
                    cluster_all_ms = st.sidebar.checkbox('Cluster All', value=True)
                    n_jobs_ms = st.sidebar.number_input('Number of Jobs', min_value=1, value=None)
                    max_iter_ms = st.sidebar.number_input('Max Iterations', min_value=1, value=300)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Mean Shift Model'):
                        mean_shift_model = MeanShift(bandwidth=bandwidth_ms, seeds=seeds_ms, bin_seeding=bin_seeding_ms, min_bin_freq=min_bin_freq_ms,
                                         cluster_all=cluster_all_ms, n_jobs=n_jobs_ms, max_iter=max_iter_ms)
                        labels_ms = mean_shift_model.fit_predict(X)
                        # Visualize clustering results
                        fig_ms = plt.figure()
                        plt.scatter(X[:, 0], X[:, 1], c=labels_ms, cmap='viridis')
                        plt.title('Mean Shift Clustering')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        st.pyplot(fig_ms)

                elif cluster_type == 'Spectral Clustering':
                    n_clusters_sc = st.sidebar.number_input('Number of Clusters', min_value=2, value=8)
                    eigen_solver_sc = st.sidebar.selectbox('Eigen Solver', [None, 'arpack', 'lobpcg', 'amg'])
                    n_components_sc = st.sidebar.number_input('Number of Components', min_value=2, value=None)
                    random_state_sc = st.sidebar.number_input('Random State', value=None)
                    n_init_sc = st.sidebar.number_input('Number of Initializations', min_value=1, value=10)
                    gamma_sc = st.sidebar.number_input('Gamma', value=1.0)
                    affinity_sc = st.sidebar.selectbox('Affinity', ['rbf', 'nearest_neighbors'], index=0)
                    n_neighbors_sc = st.sidebar.number_input('Number of Neighbors', min_value=1, value=10)
                    eigen_tol_sc = st.sidebar.text_input('Eigen Tolerance', value='auto')
                    assign_labels_sc = st.sidebar.selectbox('Assign Labels', ['kmeans', 'discretize'], index=0)
                    degree_sc = st.sidebar.number_input('Degree', min_value=1, value=3)
                    coef0_sc = st.sidebar.number_input('Coef0', value=1)
                    kernel_params_sc = st.sidebar.text_input('Kernel Parameters', value=None)
                    n_jobs_sc = st.sidebar.number_input('Number of Jobs', min_value=1, value=None)
                    verbose_sc = st.sidebar.checkbox('Verbose', value=False)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Spectral Clustering Model'):
                        spectral_clustering_model = SpectralClustering(n_clusters=n_clusters_sc, eigen_solver=eigen_solver_sc,
                                                           n_components=n_components_sc, random_state=random_state_sc,
                                                           n_init=n_init_sc, gamma=gamma_sc, affinity=affinity_sc,
                                                           n_neighbors=n_neighbors_sc, eigen_tol=eigen_tol_sc,
                                                           assign_labels=assign_labels_sc, degree=degree_sc,
                                                           coef0=coef0_sc, kernel_params=kernel_params_sc,
                                                           n_jobs=n_jobs_sc, verbose=verbose_sc)
                        labels_sc = spectral_clustering_model.fit_predict(X)
            # Visualize clustering results
                        fig_sc = plt.figure()
                        plt.scatter(X[:, 0], X[:, 1], c=labels_sc, cmap='viridis')
                        plt.title('Spectral Clustering')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        st.pyplot(fig_sc)

                elif cluster_type == 'Agglomerative':
                    n_clusters_ac = st.sidebar.number_input('Number of Clusters', min_value=2, value=2)
                    metric_ac = st.sidebar.selectbox('Metric', ['euclidean', 'manhattan', 'cosine', 'precomputed'], index=0)
                    memory_ac = st.sidebar.text_input('Memory', value=None)
                    connectivity_ac = st.sidebar.text_input('Connectivity', value=None)
                    compute_full_tree_ac = st.sidebar.selectbox('Compute Full Tree', ['auto', True, False], index=0)
                    linkage_ac = st.sidebar.selectbox('Linkage', ['ward', 'complete', 'average', 'single'], index=0)
                    distance_threshold_ac = st.sidebar.text_input('Distance Threshold', value=None)
                    compute_distances_ac = st.sidebar.checkbox('Compute Distances', value=False)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train Agglomerative Clustering Model'):
                        agglomerative_clustering_model = AgglomerativeClustering(n_clusters=n_clusters_ac, metric=metric_ac,
                                                                     memory=memory_ac, connectivity=connectivity_ac,
                                                                     compute_full_tree=compute_full_tree_ac, linkage=linkage_ac,
                                                                     distance_threshold=distance_threshold_ac, compute_distances=compute_distances_ac)
                        labels_ac = agglomerative_clustering_model.fit_predict(X)
            # Visualize clustering results
                        fig_ac = plt.figure()
                        plt.scatter(X[:, 0], X[:, 1], c=labels_ac, cmap='viridis')
                        plt.title('Agglomerative Clustering')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        st.pyplot(fig_ac)

                elif cluster_type == 'DBSCAN':
                    eps_db = st.sidebar.number_input('Epsilon', value=0.5)
                    min_samples_db = st.sidebar.number_input('Min Samples', min_value=1, value=5)
                    metric_db = st.sidebar.selectbox('Metric', ['euclidean', 'manhattan', 'cosine', 'precomputed'], index=0)
                    metric_params_db = st.sidebar.text_input('Metric Parameters', value=None)
                    algorithm_db = st.sidebar.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
                    leaf_size_db = st.sidebar.number_input('Leaf Size', min_value=1, value=30)
                    p_db = st.sidebar.text_input('P', value=None)
                    n_jobs_db = st.sidebar.number_input('Number of Jobs', min_value=1, value=None)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train DBSCAN Clustering Model'):
                        dbscan_model = DBSCAN(eps=eps_db, min_samples=min_samples_db, metric=metric_db, metric_params=metric_params_db,
                                  algorithm=algorithm_db, leaf_size=leaf_size_db, p=p_db, n_jobs=n_jobs_db)
                        labels_db = dbscan_model.fit_predict(X)
            # Visualize clustering results
                        fig_db = plt.figure()
                        plt.scatter(X[:, 0], X[:, 1], c=labels_db, cmap='viridis')
                        plt.title('DBSCAN Clustering')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        st.pyplot(fig_db)


                elif cluster_type == 'HDBSCAN':
                    min_cluster_size_hd = st.sidebar.number_input('Min Cluster Size', min_value=1, value=5)
                    min_samples_hd = st.sidebar.text_input('Min Samples', value=None)
                    cluster_selection_epsilon_hd = st.sidebar.number_input('Cluster Selection Epsilon', value=0.0)
                    max_cluster_size_hd = st.sidebar.text_input('Max Cluster Size', value=None)
                    metric_hd = st.sidebar.selectbox('Metric', ['euclidean', 'manhattan', 'cosine', 'precomputed'], index=0)
                    metric_params_hd = st.sidebar.text_input('Metric Parameters', value=None)
                    alpha_hd = st.sidebar.number_input('Alpha', value=1.0)
                    algorithm_hd = st.sidebar.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
                    leaf_size_hd = st.sidebar.number_input('Leaf Size', min_value=1, value=40)
                    n_jobs_hd = st.sidebar.number_input('Number of Jobs', min_value=1, value=None)
                    cluster_selection_method_hd = st.sidebar.selectbox('Cluster Selection Method', ['eom', 'leaf'], index=0)
                    allow_single_cluster_hd = st.sidebar.checkbox('Allow Single Cluster', value=False)
                    store_centers_hd = st.sidebar.text_input('Store Centers', value=None)
                    copy_hd = st.sidebar.checkbox('Copy', value=False)

                    st.sidebar.subheader('Train Model')
                    if st.sidebar.button('Train HDBSCAN Clustering Model'):
                        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size_hd, min_samples=min_samples_hd,
                                    cluster_selection_epsilon=cluster_selection_epsilon_hd,
                                    max_cluster_size=max_cluster_size_hd, metric=metric_hd,
                                    metric_params=metric_params_hd, alpha=alpha_hd, algorithm=algorithm_hd,
                                    leaf_size=leaf_size_hd, n_jobs=n_jobs_hd, cluster_selection_method=cluster_selection_method_hd,
                                    allow_single_cluster=allow_single_cluster_hd, store_centers=store_centers_hd, copy=copy_hd)
                        labels_hd = hdbscan_model.fit_predict(X)
            # Visualize clustering results
                        fig_hd = plt.figure()
                        plt.scatter(X[:, 0], X[:, 1], c=labels_hd, cmap='viridis')
                        plt.title('HDBSCAN Clustering')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        st.pyplot(fig_hd)


if __name__ == '__main__':
    main()
