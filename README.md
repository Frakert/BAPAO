# BAPAO

**Bayesian ALD Parameter Assessment & Optimisation (BAPAO)** is an open-source Python package for extracting physical parameters of Atomic Layer Deposition (ALD) processes from growth profiles measured in Lateral High Aspect Ratio (LHAR) structures.

BAPAO implements the physics-informed Bayesian inference framework presented in:

> **Physics-Informed Bayesian Estimation of Physical Parameters for Atomic Layer Deposition**

Using experimental growth profiles, BAPAO can estimate:

* Sticking probability (*s₀*)
* Recombination probability (*r*)
* Dimensionless reactant dose (*k*)
* Optional non-ideal ALD parameters, including desorption and non-self-limiting (CVD-like) growth

The package combines a fast analytical double-exponential fit with Bayesian Markov Chain Monte Carlo (MCMC) inference to obtain parameter estimates together with calibrated uncertainty intervals.

<!-- <p align="center">
  <img src="https://github.com/Frakert/BAPAO/blob/main/assets/BAPAO%20Logo.png" width="600" alt="BAPAO Logo">
</p> -->

---

## Features

* Bayesian inference of ALD process parameters
* Physics-informed MCMC sampling
* Double-exponential fitting of LHAR saturation profiles
* Simultaneous fitting of multiple experimental observations
* Support for thermal ALD and PEALD
* Uncertainty quantification through posterior distributions
* Extensible framework for additional ALD process models

---

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/Frakert/BAPAO.git
cd BAPAO
conda env create -f environment.yml
conda activate bapao
```

This installs all required dependencies and sets up a dedicated Conda environment.

---

## Examples

The `examples/` directory contains example workflows demonstrating:

* Double-exponential fitting
* Bayesian parameter estimation with MCMC
* Analysis of experimental growth profiles
* Joint fitting of multiple observations
* Advanced inference examples

These examples provide a good starting point for applying BAPAO to your own ALD data.

---

## Citation

If you use BAPAO in your research, please cite the accompanying publication:

Paper: [Publication link coming soon]

@article{placeholder,
    title = {},
    author = {},
    journal = {},
    year = {},
    doi = {},
    url = {}
}

---

## Contributing

Contributions are very welcome.

Whether you would like to:

* report a bug,
* request a feature,
* improve the documentation,
* contribute code, or
* discuss new ideas,

please open an issue or submit a pull request.

---

## License

See the `LICENSE` file for licensing information.
