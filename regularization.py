# Please, compare and analyze results. Add conclusions as comments here or to a readme file.

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test]

def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.
    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)

def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.
    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)

def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        BaseEstimator: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    model = Ridge()
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    model = Lasso()
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(X, y)
    return model

def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

if __name__ == "__main__":
    # Regression evaluation
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = get_regression_data()
    lin = linear_regression(X_train_reg, y_train_reg)
    rid = ridge_regression(X_train_reg, y_train_reg)
    las = lasso_regression(X_train_reg, y_train_reg)

    print("Regression Results:")
    print("Linear MSE:", mean_squared_error(y_test_reg, lin.predict(X_test_reg)))
    print("Linear R2:", r2_score(y_test_reg, lin.predict(X_test_reg)))
    print("Ridge MSE:", mean_squared_error(y_test_reg, rid.predict(X_test_reg)))
    print("Ridge R2:", r2_score(y_test_reg, rid.predict(X_test_reg)))
    print("Lasso MSE:", mean_squared_error(y_test_reg, las.predict(X_test_reg)))
    print("Lasso R2:", r2_score(y_test_reg, las.predict(X_test_reg)))
    print("Linear coefficients:", lin.coef_)
    print("Ridge coefficients:", rid.coef_)
    print("Lasso coefficients:", las.coef_)

    # Classification evaluation
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = get_classification_data()
    log = logistic_regression(X_train_clf, y_train_clf)
    log2 = logistic_l2_regression(X_train_clf, y_train_clf)
    log1 = logistic_l1_regression(X_train_clf, y_train_clf)

    print("\nClassification Results:")
    print("Logistic Accuracy:", accuracy_score(y_test_clf, log.predict(X_test_clf)))
    print("Logistic L2 Accuracy:", accuracy_score(y_test_clf, log2.predict(X_test_clf)))
    print("Logistic L1 Accuracy:", accuracy_score(y_test_clf, log1.predict(X_test_clf)))
    print("Logistic coefficients:", log.coef_[0])
    print("Logistic L2 coefficients:", log2.coef_[0])
    print("Logistic L1 coefficients:", log1.coef_[0])
