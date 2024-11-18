# Kernel Banzhaf
This repo implements ```KernelBanzhaf``` (with and without paired sampling), ```MC```, and ```MSR``` estimators for estimating Banzhaf values in feature attribution; it also provides two ways of feature perturbation: interventional (```RawImputer```) or tree path dependent (```TreeImputer```)

### Python setup 

```console
$ conda create --name {env-name}
$ conda activate {env-name}
$ pip install -r requirements.txt
```

### Usage
To execute the code, you must: 1) load the baseline data, the explicand, and the model you intend to explain; 2) execute your preferred Banzhaf value estimator. For instance, you can employ the Kernel Banzhaf algorithm as follows:
```python
import shap  # Used for loading datasets
import xgboost as xgb  # Importing XGBoost for the model

from imputer import RawImputer
from explainers import KernelBanzhaf
from utils import get_data_and_explicand

# Load the data
X, y = shap.datasets.adult()
features = X.columns.tolist()

# Choose baseline data and the explicand
baseline, explicand = get_data_and_explicand(data_size=50, base_data=X)

# Load and train the model
model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
model.fit(X, y)

# Use interventional feature perturbation (raw imputer)
imputer = RawImputer(baseline, explicand, features, model)

# Use Kernel Banzhaf with paired sampling to estimate the Banzhaf values
sample_size = 1000
banzhaf_values = KernelBanzhaf(features, sample_size, imputer)()
```