Convergence Analysis for MLSDC
==============================

This project investigates the impact of the convergence results described in the paper "Convergence Analysis for Multi-level Spectral Deferred Corrections".
We use the 1D heat equation as well as the 2D Allen Cahn equation to show how MLSDC converges for a linear/non-linear PDE.

Test equations
--------------

- ``solve_heat1d.py``: MLSDC is applied to the 1D heat equation with different method parameters. In particular, the tests include an example with optimal parameter selection (according to theorem 5 in the paper) and three examples with individual changes from this resulting in a lower convergence order. The results are saved in individual files in the data folder.
- ``solve_allencahn._2dpy``: MLSDC is applied to the 2D Allen Cahn equation with different method parameters. As for the heat equation, the tests include an example with optimal parameter selection (according to theorem 5 in the paper) and three examples with individual changes from this resulting in a lower convergence order. The results are saved in individual files the data folder.

Other files
-----------

- ``plot_errors.py``: Plots the results of a single test example. Both SDC and MLSDC errors are graphically illustrated and can be compared.
- ``plot_errors_mlsdc_variants.py``: Plots the results of various examples in one figure. Only the MLSDC errors are graphically illustrated and compared.
- ``sweeper_random_initial_guess``: Inherited from the core sweeper class. The possibility to use a random initial guess is added.

How to
------

- execute ``solve_heat1d.py`` or ``solve_allencahn_2d.py`` to generate the results (fig numbers 1 to 4 are the ones illustrated in the paper, with fig=0 you may also choose other parameters and have a look at the results)
- execute ``plot_errors_mlsdc_variants.py`` to generate the corresponding plots appearing in the paper (choose the desired IVP at the beginning of the the main-function, just comment or uncomment the respective line)
