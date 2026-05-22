import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import ElasticNet
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder
import time
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv('DataSets/pitcher_data_cleaned.csv')
df = df[df['pitch_type']!='PO']
le = LabelEncoder()

X = df.drop(columns=['pitch_type', 'plate_x', 'plate_z'])
y_pitch_type = le.fit_transform(df['pitch_type'])
y_plate_x = df[['plate_x']]
y_plate_z = df[['plate_z']]

encoded_to_original = dict(zip(le.transform(le.classes_), le.classes_))

start = time.time()
preprocessor = ColumnTransformer([
    ('dummify', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['pitcher', 'batter', 'on_1b', 'on_2b', 'on_3b', 'if_fielding_alignment', 'of_fielding_alignment', 'prev_pitch_type']),
    ("standardize", StandardScaler(), ['inning', 'balls', 'strikes', 'outs_when_up', 'score_diff'])
], remainder='drop')

y_series = pd.Series(y_pitch_type)

classes = np.unique(y_pitch_type)
class_weights = compute_class_weight('balanced', classes=classes, y=y_pitch_type)
class_weight_dict = dict(zip(classes, class_weights))

sample_weights = y_series.map(class_weight_dict).values

XGB = XGBClassifier(n_jobs=-1)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('boosting', XGB)
])

param_grid = {
    'boosting__max_depth': [4], 'boosting__n_estimators': [75], 'boosting__learning_rate': [.40]}

stratified_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_cv, scoring='balanced_accuracy', n_jobs=-1)

grid_search.fit(X, y_pitch_type, boosting__sample_weight=sample_weights)
end = time.time()

print(f"Execution time: {end - start:.2f} seconds")
print("Best accuracy score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)

pitch_type_dict = {'ST':'Sweeper', 'CH': "Changeup", 'FF': "Four Seam Fastball", 'SI': "Sinker", 'SL': "Slider", 'FC': "Cutter", 'CU': "Curveball", 'KC': "Knuckleball", 'FS': "Split-finger"}


joblib.dump(grid_search.best_estimator_, "best_pitch_model.pkl")

model = joblib.load('best_pitch_model.pkl')

