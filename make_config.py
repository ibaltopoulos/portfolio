
def write_yaml(f, d, indent = 0):
    for key, value in d.items():
        f.write(" "*indent + key + ":")
        if isinstance(value, dict):
            f.write("\n")
            write_yaml(f, value, indent + 2)
        else:
            f.write(" " + str(value))
            f.write("\n")
    f.write("\n")



def write_config():

    d = {}
    d["estimator"] = {"pickle": "model.pkl"}
    d["strategy"] = {"name": "gp", "seeds": "10", "acquisition": "{name':'osprey', 'params': {}}"}
    d["search_space"] = {"n_hidden": {"min": "1", "max": "300", "type": "int", "warp": "log"},
        "l1_reg": {"min": "1e-7", "max": "1e-1", "type": "float", "warp": "log"},
        "l2_reg": {"min": "1e-7", "max": "1e-1", "type": "float", "warp": "log"},
        "learning_rate": {"min": "1e-4", "max": "1e1", "type": "float", "warp": "log"}
        }
    d["cv"] = {"name": "kfold", "params": {"n_splits": 3, "shuffle": "False"}}
    d["dataset_loader"] = {"name": "joblib", "params": {"filenames": "data.pkl", "x_name": "x", "y_name": "y"}}
    d["trials"] = "sqlite:///osprey-trials.db"

    with open("test.yaml", "w") as f:
        write_yaml(f, d)


if __name__ == "__main__":
    MODEL_NAME = "model.pkl"
    STRATEGY = "gp"
    ACQUISITION = "osprey"

    write_config()



