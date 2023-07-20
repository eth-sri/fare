# FARE <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

The code accompanying our ICML 2023 paper: [**FARE: Provably Fair Representation Learning with Practical Certificates**](https://www.sri.inf.ethz.ch/publications/jovanovic2023fare).

## Code overview 

The directory `code/` contains implementations of FARE and all baselines. We use the library of [Gupta et al. (2021)](https://github.com/umgupta/fairness-via-contrastive-estimation) as a starting point, including their implementation of FCRL and CVIB baselines. These baselines use config files from `code/config`, files in `code/lib`, `code/src/arch`, `code/src/models` and `code/src/trainers`, and their entry point is `code/src/scripts/main.py`. For other baselines (LAFTR, FNF, fair-path, and sIPM) we use the corresponding official repositories, that we place in corresponding folders in `code/src`. Noop is the baseline that uses the original data as representations. Finally, `code/src/common` holds metrics and the code to load the datasets.

### FARE code

The high-level parts of FARE implementation are in `code/src/tree/`. This includes the data preprocessing and postprocessing and the fairness proof, and invokes 
our implementation of fair decision tree learning, implemented as a patch for `sklearn` which augments the `DecisionTreeClassifier` class. Thus, the low-level code of the fair decision tree is in `sktree/`. To patch `sklearn` (when setting up for the first time, or when changing something in `sktree`), it is required to have a copy of sklearn in `scikit-learn/` in the project root, corresponding to [this commit](https://github.com/scikit-learn/scikit-learn/commit/fd60379f95f5c0d3791b2f54c4d070c0aa2ac576). Then, running `sktree/build.sh` installs the patch and should enable running FARE.

The code in `sktree` is written in Cython. Here, `.pxd` files roughly correspond to C++ `.hpp` files, and `.pyx` files correspond to C++ `.cpp` files. The 2 key places in this code are in `_criterion.pyx` (L631), where we define the `FairGini` criterion---a tree splitting criterion that takes into account both accuracy and fairness, and `_splitter.pyx` (L467), where we deal with ordinally encoded categorical variables, instead of the common 1-hot encoding which is the default in `sklearn`. Both of these are described in more detail in the paper. Other than these 2 places, adding a new parameter to the tree requires more plumbing, for this refer to past commits that did the same.

## Requirements

We require conda and latex preinstalled. The conda environment is given in `fare_env.yml`. Running `install.sh` creates it from the `yml` and applies the cloning and patching of sklearn as described above. This should be sufficient to reproduce our results.

## Running the code 
Assume `code/` as the working directory. Running the code (to e.g., reproduce our main results in Figures 3, 4 and 10) is divided in three main steps.

**1) Running a method to produce representations.**
For given dataset and method, use the script `shell/<dataset>/run_<method>.sh`. This runs the corresponding FRL method with various parameter sets and stores the resulting representations locally in `result/<dataset>/<method>/<paramstring>/embeddings.npy` (or similar). 

**2) Evaluating the representations on a downstream task.**
See the full example in `shell/eval_embeddings_example.sh`. Run `src/scripts/eval_embeddings.py` first, using the corresponding config from `config/` and the `--key` parameter to parallelize evaluation. For example `--key 1%3` will list all embeddings from the folder indicated by `-f` and run only those at indices `3k+1`. Each run will produce a `result/_eval/<dataset>/<method>.npy` result file, containing accuracy, fairness, and the fairness bound (if FARE). If using `--key` these should be merged using `src/scripts/merge_embeddings.py`.

**3) Plotting the results**
For this, use `python3 -m src.scripts.plot -p` (`p` stands for pareto fronts), which loads the result `.npy` files from `result/_eval` and saves the plots to `plots/`. See the bottom of `plot.py` to enable/disable some plots.

### Data from experiments used in the paper 
All merged `.npy` result files used in our experiments in the paper are present in the repository. This means that it is not necessary to rerun everything to produce the plots. For example, for Figure 3, the results are in `result/_eval` under directories `ACSIncome-CA-2014`, `ACSIncome-ALL-2014`. See `plot.py` for the exact files used for each plot. Notably, the scaling experiment (Figure 5) data is provided in `result/_eval/Ns.npy`, and the FARE certificate validity experiment (Figure 6) data is provided in `result/_eval/tree_health_notransfer.npy`.

## Citation

```
@article{
    jovanovic2023fare,
    title={FARE: Provably Fair Representation Learning with Practical Certificates},
    author={Jovanovi{\'c}, Nikola and Balunovi{\'c}, Mislav and Dimitrov, Dimitar I and Vechev, Martin},
    booktitle={Proceedings of the 40th International Conference on Machine Learning},
    year={2023},
    publisher={PMLR}
}
```
