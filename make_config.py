
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
    if STRATEGY == "sobol":
        d["strategy"] = {"name": STRATEGY}
    elif STRATEGY == "hyperopt_tpe":
        d["strategy"] = {"name": STRATEGY, "params": {"seeds": SEEDS}}
    elif KAPPA == None:
        d["strategy"] = {"name": STRATEGY, "params": {"seeds": SEEDS, "acquisition": "{ name : %s, params : {}}" % ACQUISITION, 
                "n_init": NUM_INIT, "n_iter": NUM_ITER, "sobol_init": SOBOL_INIT}}
    else:
        d["strategy"] = {"name": STRATEGY, "params": {"seeds": SEEDS, 
                "acquisition": "{ name : %s, params : {'kappa': %s}}" % (ACQUISITION, KAPPA), 
                "n_init": NUM_INIT, "n_iter": NUM_ITER, "sobol_init": SOBOL_INIT}}


    d["search_space"] = {
        "l1_reg": {"min": "1e-7", "max": "1e-1", "type": "float", "warp": "log"},
        "l2_reg": {"min": "1e-7", "max": "1e-1", "type": "float", "warp": "log"},
        "learning_rate": {"min": "1e-4", "max": "1e1", "type": "float", "warp": "log"}
        }
    if STRATEGY == "hyperopt_tpe":
        d["search_space"]["n_hidden"] = {"min": "1", "max": "300", "type": "int"}
    else:
        d["search_space"]["n_hidden"] = {"min": "1", "max": "300", "type": "int", "warp": "log"}

    d["cv"] = {"name": "kfold", "params": {"n_splits": "3", "shuffle": "False"}}
    d["dataset_loader"] = {"name": "joblib", "params": {"filenames": "data.pkl", "x_name": "x", "y_name": "y"}}
    d["trials"] = {"uri": "sqlite:///trials_%d.db" % COUNTER}

    with open("config%d.yaml" % COUNTER, "w") as f:
        write_yaml(f, d)


if __name__ == "__main__":
    SEEDS = None
    KAPPA = None
    NUM_ITER = None
    NUM_INIT = None
    SOBOL_INIT = None
    ACQUISITION = None
    COUNTER = 1

    STRATEGY = "sobol"
    write_config()
    SEEDS = 10
    COUNTER += 1
    STRATEGY = "hyperopt_tpe"
    write_config()

    STRATEGY = "gp"
    for ACQUISITION in ["osprey", "ei", "ucb", "lars"]:
        for KAPPA in [None, "1", "2", "3"]:
            if ACQUISITION != "ucb" and KAPPA != None:
                continue
            if ACQUISITION == "ucb" and KAPPA == None:
                continue
            for NUM_ITER in ["1", "10", "100"]:
                for NUM_INIT in ["1", "10", "100"]:
                    for SOBOL_INIT in ["True", "False"]:
                        if SOBOL_INIT == "True" and NUM_ITER == "1":
                            continue
                        COUNTER += 1
                        #if ACQUISITION == "ucb":
                        #    print(KAPPA, COUNTER)
                        write_config()



