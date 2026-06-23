import importlib, sys
mods = ['xgboost', 'optuna', 'joblib', 'sklearn', 'numpy', 'pandas']
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m}:OK")
    except Exception as e:
        print(f"{m}:ERR:{e}")
