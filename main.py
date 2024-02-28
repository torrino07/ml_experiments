import pandas as pd
import histfuncs as hf
from sklearn.linear_model import LinearRegression
import numpy as np

file_path = "data/assignment_train_sample.csv"

df = pd.read_csv(file_path)

# Preprocess the Data {0,1} -> {−1,1}. Apply 2*y - 1 to output columns
columns_to_transform = ["P1 y","P2 y","P3 y","P4 y"]
for col in columns_to_transform:
    df[col] = df[col].apply(lambda x: 2*x - 1)


# Drop column
df = df.drop("Unnamed: 0", axis=1)

# Question 7
for i in range(1, 5):
    Y = df[f"P{i} y"].values
    X = df[[f"P{i} x1", f"P{i} x2"]].values
    
    x1_max, x1_min = max(X[:, 0]), min(X[:, 0])
    x2_max, x2_min = max(X[:, 1]), min(X[:, 1])
    
    # # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, Y)
    
    # define parameters
    w1, w2 = model.coef_
    b = model.intercept_
    
    # predictions
    pred_func = hf.pred_func_simple_linear_model(w1, w2, b)
    
    # Calculate predictions for the entire dataset
    predictions = np.array([pred_func(x) for x in X])

    # Compute the empirical risk using the surrogate loss function
    surrogate_losses = (Y - predictions) ** 2
    empirical_risk = np.mean(surrogate_losses)
    print(f"Problem {i}: {empirical_risk}")
    
    # plot
    hf.plot_dec_bound(h=0.001, predict_func=pred_func, x1min=x1_min, x1max=x1_max, x2min=x2_min,
                  x2max=x2_max, X=X, y=Y, title=f"Problem {i}", label_1="x1", label_2="x2")

# Question 8
for i in range(1, 5):
    Y = df[f"P{i} y"].values
    X = df[[f"P{i} x1", f"P{i} x2"]].values
    
    x1_max, x1_min = max(X[:, 0]), min(X[:, 0])
    x2_max, x2_min = max(X[:, 1]), min(X[:, 1])
    
    X_interaction = X[:, 0] * X[:, 1]
    X21 = X[:, 0] ** 2
    X22 = X[:, 1] ** 2
    
    # Extend the model to include interaction terms and quadratic trasnformations of features
    X_extended = np.c_[X, X_interaction, X21, X22]
    
    # # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_extended, Y)
    
    # define parameters
    m1, m2, m3, m4, m5 = model.coef_
    b = model.intercept_
    
    # predictions
    pred_func = hf.pred_func_flexible_linear_model(m1, m2, m3, m4, m5, b)
    
    # Calculate predictions for the entire dataset
    predictions = np.array([pred_func(x) for x in X])

    # Compute the empirical risk using the surrogate loss function
    surrogate_losses = (Y - predictions) ** 2
    empirical_risk = np.mean(surrogate_losses)
    print(f"Problem {i}: {empirical_risk}")
    
    # plot
    hf.plot_dec_bound(h=0.001, predict_func=pred_func, x1min=x1_min, x1max=x1_max, x2min=x2_min,
                  x2max=x2_max, X=X, y=Y, title=f"Problem {i}", label_1="x1", label_2="x2")