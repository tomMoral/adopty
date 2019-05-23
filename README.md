# Code to reproduce the NeurIPS submission : "_Learning step sizes for unfolded sparse coding_"



## Compat

This package has been developed and tested with `python3.6`. It is therefore not guaranteed to work with earlier versions of python.

## Install the repository on your machine


This package can easily be installed using `pip`, with the following command:

```bash
pip install numpy
pip install -e .
```

This will install the package and all its dependencies, listed in `requirements.txt`. To test that the installation has been successful, you can install `pytest` and run the test suite using

```
pip install pytest
pytest
```


## Reproducing the figures of the paper

Figure 2 - Convergence curves between ISTA, FISTA and OISTA.

```bash
python examples/comparison_oracle_ista.py
```

Figure 4 - Learn a small network and display the learned step as well as the distribution of 1/L_S at each layer on the training set.

```bash
python examples/plot_learned_steps.py
```


Figure 5 - Show that when learning a LISTA network with 40 layers, the last layers look like SLISTA layers, where only the step size differs from ISTA. This verifies numerically theorem 4.4.

```bash
python examples/plot_dict_similarity.py
```


Figure 6 - comparison of ISTA, LISTA, ALISTA and SLISTA on large scale problems. Note that this experiment can take up to 24h to be generated. You can tweak the parameter N_JOB and N_GPU in `examples/run_comparison_networks.py` to accelerate the computations by parallelization and using GPU. Note that if both are set, you will have N_JOB / N_GPU jobs running on each GPU, which should be chosen reasonably.

```bash
python examples/run_comparison_networks.py
python examples/plot_comparison_networks.py
```

## Reproducing figures from the appendix


Figure E.1 - Plot the distribution of L_S / L as a function of the cardinal of S.
```bash
python examples/plot_sparsity_distribution.py
```

Figure E.2 - Plot histograms of number of iterations for ISTA, FISTA and OISTA.
```bash
python examples/run_comparison_iterative.py
python examples/plot_comparison_iterative.py
```

Figure E.3 - Plot the test loss relatively to the number of training points used to optimized LISTA or SLISTA.
```bash
python examples/plot_learning_curves.py
```
