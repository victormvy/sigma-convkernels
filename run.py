import subprocess

methods = [
            {'name' : 'hydra', 'temporal' : True},
            {'name' : 'mr', 'temporal' : True},
            {'name' : 'rocket', 'temporal' : True},
            {'name' : 'inception', 'temporal' : True},
            {'name' : 'xgb', 'temporal' : False},
            {'name' : 'rf', 'temporal' : False},
            {'name' : 'ridge', 'temporal' : False}
        ]

seeds = [2, 3, 5, 7, 11, 13]
n_folds = 5

item_data = []

for method in methods:
    for seed in seeds:
        for fold in range(n_folds):
            process = subprocess.Popen(f"python main.py with method={method['name']} temporal={str(method['temporal'])} n_folds={str(n_folds)} fold={str(fold)} seed={str(seed)}")
            process.wait()