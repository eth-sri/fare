"""Take a folder... go to all folders and look for config.json"""
import argparse
import importlib
import logging
import os
import datetime

import numpy
from box import Box
from checksumdir import dirhash
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lib.src.os_utils import try_cast, safe_makedirs
from src.common.metrics import demographic_parity_difference, equalized_odds_difference, equal_opportunity_difference, demographic_parity_difference_soft


def run_sklearn_model(config, z_train, y_train, z_test, y_test):
    """Config is dictionary that is expected to have following attr
        .name : <name of the model>
        The attr that have the same name as attr of sklearn model are passed to the model as is.
        __<name>__<attr> are caught and passed to method with name <name>
    """

    def get_options(string):
        options = {}
        for (k, v) in config.items():
            if string in k:
                options[k.split(string)[1]] = v
        return options

    try:
        if config.get("preprocess") == "normalize":
            logger.debug("Normalizing")
            model = StandardScaler()
            model.fit(z_train)
            z_train = model.transform(z_train)
            z_test = model.transform(z_test)

        if config.name == "nn":
            options = get_options("__nn__")
            logger.debug(f"Running nn with config {config}")
            model = MLPClassifier(**options, verbose=True)

        if config.name == "logreg":
            options = get_options("__logreg__")
            logger.debug(f"Running logistic regression with config {config}")
            model = LogisticRegression(**options)

        if config.name == "rf":
            options = get_options("__rf__")
            logger.debug(f"Running random forest with config {config}")
            model = RandomForestClassifier(**options)

        if config.name == "tree":
            options = get_options("__tree__")
            logger.debug(f"Running decision tree with config {config}")
            model = DecisionTreeClassifier(**options)

        if config.name == "svm":
            options = get_options("__svm__")
            logger.debug(f"Running svm with config {config}")
            if not config.get("bagging"):
                # don't use bagging
                model = SVC(probability=True, **options)
            else:
                # use bagging
                n_estimators = 10
                model = BaggingClassifier(SVC(probability=True, **options),
                                          n_estimators=n_estimators, max_samples=1 / n_estimators,
                                          bootstrap=False)
        model.fit(z_train, y_train)
        score = model.score(z_test, y_test)
        prob = model.predict_proba(z_test)
        logger.debug(f"score train: {model.score(z_train, y_train)}")

        return score, prob

    except Exception as e:
        logger.debug(f"Error occured while running sklearn model {e}")
        return None, None


def evaluate_embeddings(folder, target_evaluations, method, existing_result=None, transfer_label_idx=None):
    if os.path.exists(os.path.join(folder, "embeddings.npy")):
        embedding_path = os.path.join(folder, "embeddings.npy")
    elif os.path.exists(os.path.join(folder, "embedding.npy")):
        embedding_path = os.path.join(folder, "embedding.npy")
    else:
        logger.error(f"Embeddings not found in {folder}, dropping to shell")
        import code; code.interact(local=dict(globals(), **locals()))
        return None

    logger.info(f"Loading embedding from {embedding_path}")
    data = numpy.load(embedding_path, allow_pickle=True).item()
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(embedding_path))
    print(f'Modified on: {mtime}')

    if method in ['fnf', 'sipm', 'fair-path', 'kmeans']:
        # temporary fix 
        for k in ['z_train', 'z_test', 'c_train', 'c_test', 'y_train', 'y_test']:
            print(f'type of {k}: {type(data[k])}')
            if type(data[k]) != numpy.ndarray:
                print('fixing torch -> numpy, generally this should not happen though')
                data[k] = data[k].cpu().numpy()

    # TODO consolidate all these 
    if method == "laftr":
        z_train = numpy.concatenate([data["train"]["Z"], data["valid"]["Z"]], axis=0)
        y_train = numpy.concatenate([data["train"]["Y"], data["valid"]["Y"]], axis=0)
        c_train = numpy.concatenate([data["train"]["A"], data["valid"]["A"]], axis=0)

        z_test = data["test"]["Z"]
        y_test = data["test"]["Y"]
        c_test = data["test"]["A"]
    elif method in ['tree', 'noop', 'fnf', 'sipm', 'fair-path', 'kmeans', 'tree-eo', 'tree-eopp']:
        z_train = data["z_train"].astype(numpy.float64)
        c_train = data["c_train"]
        y_train = data["y_train"]

        z_test = data["z_test"].astype(numpy.float64)
        c_test = data["c_test"]
        y_test = data["y_test"]

    elif method == "lag-fairness":
        z_train = data["train"]["z"]
        y_train = data["train"]["y"]
        c_train = data["train"]["u"]
        if len(c_train.shape) == 2 and c_train.shape[1] > 1:
            c_train = numpy.argmax(c_train, axis=1)

        z_test = data["test"]["z"]
        y_test = data["test"]["y"]
        c_test = data["test"]["u"]
        if len(c_test.shape) == 2 and c_test.shape[1] > 1:
            c_test = numpy.argmax(c_test, axis=1)

    elif method == "adv_forgetting":
        z_train = data["z_wave_train"]
        c_train = data["c_train"]
        y_train = data["y_train"]

        z_test = data["z_wave_test"]
        c_test = data["c_test"]
        y_test = data["y_test"]

    else:
        z_train = data["z_train"]
        c_train = data["c_train"]
        y_train = data["y_train"]

        z_test = data["z_test"]
        c_test = data["c_test"]
        y_test = data["y_test"]

    if existing_result is not None:
        result = existing_result
    else:
        result = {}

    # sanity check that test set is the same
    print(y_test.shape)
    print(str(y_test[10:20]))

    if  len(y_train.shape) > 1 and y_train.shape[1] > 1:
        assert transfer_label_idx is not None 
        print(f'Transferring to label {transfer_label_idx}')
        y_train = y_train[:, transfer_label_idx]
        y_test = y_test[:, transfer_label_idx]
    else:
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)


    # target evaluation here
    if target_evaluations is not None:
        N = target_evaluations.get("num_runs", 1)
        logger.debug(f"Running {N} runs with each model")
        print("", flush=True)

        targets = [('y', y_train, y_test)]
        if target_evaluations.get("adversaries") is not None:
            targets.append(('s', c_train, c_test))
            targets = list(reversed(targets))
        
        print(f'There are {len(targets)} targets!')

        for ts in targets:
            target_name, target_train, target_test = ts[0], ts[1], ts[2]
            print(f'\n\n\nNEW TARGET: {target_name}\n\n\n')

            for config in target_evaluations.model_config:
                if result.get(config.friendly_name) is not None:
                    acc = result[config.friendly_name]["acc"]
                    auc = result[config.friendly_name]["auc"]
                    auc_micro = result[config.friendly_name]["auc_micro"]
                    dp = result[config.friendly_name]["dp"]
                    dp_soft = result[config.friendly_name]["dp_soft"]
                    logger.debug(f"Found {len(acc)} runs")
                else:
                    acc, pred, auc, auc_micro, dp, dp_soft = [], [], [], [], [], []
                    eo, eopp = [], []

                for i in range(N - len(acc)):
                    logger.debug(f"Run: {i}")
                    acc_, prob_ = run_sklearn_model(config, z_train, target_train, z_test, target_test)
                    if acc_ is None:
                        logger.debug("Accuracy was None")
                        continue
                    dp_, _ = demographic_parity_difference(target_test, c_test, prob_)
                    eo_, _ = equalized_odds_difference(target_test, c_test, prob_)
                    eopp__, _ = equal_opportunity_difference(target_test, c_test, prob_)
                    #dp_soft_, _ = demographic_parity_difference_soft(y_test, c_test, prob_)
                    #auc_ = roc_auc_score(y_test, prob_[:, 1])
                    #l = len(y_test)
                    #onehot = numpy.zeros((l, 2))
                    #onehot[numpy.arange(l), y_test.astype(int)] = 1
                    #auc_micro_ = roc_auc_score(onehot, prob_, average="micro")

                    print(f'********** Run {i} done: acc={acc_:.3f}, dp={dp_:.3f}\n', flush=True)
                    acc.append(acc_)
                    dp.append(dp_)
                    eo.append(eo_)
                    eopp.append(eopp__)
                    #dp_soft.append(dp_soft_)
                    #auc.append(auc_)
                    #auc_micro.append(auc_micro_)
                    #pred.append((prob_, y_test))

                if len(acc) > 0:
                    friendly_name = config.friendly_name 
                    
                    if target_name == 's':
                        friendly_name += '_adversary'

                    result[friendly_name] = {"acc": acc, "auc": auc, "dp": dp, "eo": eo, "eopp": eopp,
                                                    "dp_soft": dp_soft,
                                                    # "pred": pred,
                                                    "auc_micro": auc_micro}
                    for fm in ['dp', 'eo', 'eopp']:
                        if f'{fm}_ub' in data:
                            result[friendly_name][f'{fm}_ub'] = data[f'{fm}_ub']
                            print(f'there is UB on {fm}!')
    print('Result returned:')
    print(result)
    return result


def get_param_from_string(param_folder, method):
    params = {}

    if method == "lag-fairness":
        for name, p in zip(["mi", "e1", "e2", "e3", "e4", "e5"], param_folder.split("-")):
            params[name] = try_cast(p)
        logger.debug(params)
    elif method == "laftr":
        param_folder = ".".join(param_folder.split(".")[:-1])
        # just read the number after g
        param = param_folder.split("_")[-1]
        params["g"] = try_cast(param)
    elif method in ['fnf', 'sipm', 'fair-path', "tree", 'kmeans', 'tree-eo', 'tree-eopp']:
        for param in param_folder.split(","):
            if "=" in param:
                params[param.split("=")[0]] = try_cast(param.split("=")[1])
    else:
        for param in param_folder.split("_"):
            if "=" in param:
                params[param.split("=")[0]] = try_cast(param.split("=")[1])
    return params


if __name__ == "__main__":
    # define logger etc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default='0%1')
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--force", action='store_true', help="force evaluate everything")
    parser.add_argument("-c", "--config", required=True, help="config to evaluate embeddings")

    parser.add_argument("-f", "--folder_path", required=True)
    parser.add_argument("-r", "--result_folder", required=True)
    parser.add_argument("-e", "--evals", type=int, default=None)
    parser.add_argument("--transfer_label_idx", type=int, default=None) # which label to eval on? 

    parser.add_argument(
        "-m", "--method", required=True,
        help="some methods might need special treatment. This is for that and for naming"
    )

    args = parser.parse_args()

    suffix = f'-transfer-{args.transfer_label_idx}' if args.transfer_label_idx is not None else ''
    save_path = f"{args.result_folder}/{args.method}{suffix}_{args.key}.npy"

    configFile = args.config
    spec = importlib.util.spec_from_file_location("config", configFile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target_evaluations = module.target_evaluations
    if args.evals:
        module.target_evaluations.num_runs = args.evals


    if args.debug:
        logger.setLevel(logging.DEBUG)

    # load already evaluated things. if force is true... we evaluate everything again
    if os.path.exists(save_path) and args.force is False:
        logger.debug("Found existing evaluations .. using that")
        evaluations = numpy.load(save_path, allow_pickle=True).item()
    else:
        evaluations = Box({})  # result array
        evaluations.checksums = Box({})

    drs = list(sorted(os.listdir(args.folder_path)))

    aaa,bbb = args.key.split('%')
    aaa = int(aaa)
    bbb = int(bbb)

    for i, param_folder in enumerate(drs):
        print(f'{i}/{len(drs)}: {param_folder}', flush=True)
        if i % bbb != aaa:
            print('skipping', flush=True)
            continue 
        
        print('doing', flush=True)

        # if evaluations exists and checksum matches... we can use the results
        # else ... remove the results and proceed
        if evaluations.get(param_folder) is not None:
            logger.debug(f"{param_folder} already present.. so checksumming")
            # compute hash to compate
            hash = dirhash(os.path.join(args.folder_path, param_folder))
            if hash == evaluations.checksums.get(param_folder):
                logger.debug(
                    f"{param_folder} already present.. and hash matched so reusing the results")
            else:
                logger.debug(f"{param_folder} already present.. but hash mis-matched")
                evaluations.checksums[param_folder] = hash

                # empty the result
                evaluations[param_folder] = []

        # special methods with no run_<num>
        if args.method in ["laftr", "lag-fairness", "ml-arl", "tree", "noop", 'fnf', 'sipm', 'fair-path', 'kmeans', 'tree-eo', 'tree-eopp']:

            evaluations[param_folder] = []

            params = get_param_from_string(param_folder, args.method)

            r = evaluate_embeddings(os.path.join(args.folder_path, param_folder),
                                        target_evaluations=target_evaluations, method=args.method, transfer_label_idx=args.transfer_label_idx)
            if r is not None:
                # we have space for config if we need to read but we dont read now
                evaluation = Box({"result": r, "config": {},
                                  "params": params})
                evaluations[param_folder].append(evaluation)

        else:
            # there will run_<num> folder here, we will store results for all param_folders in an
            # array
            runz = sorted(os.listdir(os.path.join(args.folder_path, param_folder)))
            # (!!) just use the last run 
            run_folder = runz[-1]
            print(f'Just using the last run which is: {run_folder} !!!')

            evaluations[param_folder] = []
            # get result
            r = evaluate_embeddings(os.path.join(args.folder_path, param_folder, run_folder),
                                    target_evaluations=target_evaluations, method=args.method, transfer_label_idx=args.transfer_label_idx)
            if r is not None:
                # we have space for config if we need to read but we dont read now
                evaluation = Box({"result": r, "config": {},
                                    "params": get_param_from_string(param_folder, args.method)})
                evaluations[param_folder].append(evaluation)

    safe_makedirs(args.result_folder)
    numpy.save(save_path, evaluations)
    print('DONE.')
