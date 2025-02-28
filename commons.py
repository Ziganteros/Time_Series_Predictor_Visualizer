parameters = {
    "---": "", 
    "Linear Regression": """numero di previsioni=365 
lookback=100
""", 
    "SVR": """numero di previsioni=365
lookback=100
kernel=rbf
C=100
epsilon=0.1
""",
    "Random Forest": """numero di previsioni=365
lookback=100
n_estimators=100
criterion=squared_error
""",
    "Ridge": """numero di previsioni=365
lookback=100
alpha=1.0
""",
    "Lasso": """numero di previsioni=365
lookback=100
alpha=1.0
""",
    "Decision Tree": """numero di previsioni=365
lookback=100
max_depth=3
"""
}
