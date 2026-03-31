import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_pickle('df.pkl')
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

# Categorical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Construct pipeline
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100, random_state=42, max_samples=0.5, max_features=0.75, max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Train
pipe.fit(X, y)

# Save
pickle.dump(pipe, open('pipe.pkl', 'wb'))

print("New pipeline trained and saved successfully.")
