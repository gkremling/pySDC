Convergence Analysis for MLSDC
==============================

This project investigates the impact of the convergence results described in the paper "Convergence Analysis for Multi-level Spectral Deferred Corrections".
We use the 1D heat equation as well as the 2D Allen Cahn equation to show how MLSDC converges for a linear/non-linear PDE.

Test equations and figure creation
----------------------------------

- ``solve_heat1d.py``: Here, MLSDC is applied to the 1D heat equation with different method parameters. In particular, the tests include an example with optimal parameter selection (according to theorem 5 in the paper) and three examples with individual changes from this resulting in a lower convergence order. The results are saved in individual files in the data folder.
- ``solve_allencahn_2d.py``: Here, MLSDC is applied to the 2D Allen Cahn equation with different method parameters. As for the heat equation, the tests include an example with optimal parameter selection (according to theorem 5 in the paper) and three examples with individual changes from this resulting in a lower convergence order. The results are saved in individual files in the data folder.

- ``plot_errors.py``: Plots the results of a single test example. Both SDC and MLSDC errors are graphically illustrated and can be compared.
- ``plot_errors_mlsdc_variants.py``: Plots the results of various examples in one figure. Only the MLSDC errors are graphically illustrated and compared.

Other files
----------- 
- ``sweeper_random_initial_guess``: Inherited from the core sweeper class. The possibility to use a random initial guess is added.
- ``AllenCahn_2D_FD_sin.py``: Similar to the Allen-Cahn implementation in the implementations package but with a sine wave of arbitrary frequency as initial value.

How to
------

- execute ``solve_heat1d.py`` or ``solve_allencahn_2d.py`` to generate the numerical results for the respective test equation (fig numbers 1 to 4 are the ones illustrated in the paper, with fig=0 you may also choose other arbitrary parameters and have a look at the corresponding results)
- (afterwards) execute ``plot_errors_mlsdc_variants.py`` to generate the plots appearing in the paper (choose the desired IVP at the beginning of the the main-function by commenting/uncommenting the respective line, you can easily change the used data files, subplot titles and expected orders)
