# Quad-PH-MOR

The code accompanying my [Masters thesis](https://github.com/peoe/ma-thesis).

## Necessary Packages

The following Python packages are necessary to run the examples:
* A pyMOR version >= 2023.1 with an implementation of the pH DMD algorithm (currently only in [development branches](https://github.com/pymor/pymor/pull/2027))
* The following DUNE packages (all version 2.10):
  * dune-alugrid
  * dune-common
  * dune-fem
  * dune-geometry
  * dune-grid
  * dune-istl
  * dune-localfunctions
* The required dependencies of the above

## Experiments

To run the experiments first call the `combined_compute.py` script and thereafter `combined_evaluate.py`.
