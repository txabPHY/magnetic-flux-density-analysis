# Muon Spectroscopy Data Analysis (Python)

This project processes muon spectroscopy data to estimate magnetic flux density and analyse damping behaviour over time. It demonstrates scientific computing, signal processing, data modelling and visualisation using Python.

## Overview

Given experimental time–channel data collected at implantation energies ranging from 5 keV to 25 keV, the analysis:

* Loads and structures multi-column experimental data
* Separates detector signals by energy
* Performs modelling to estimate magnetic flux density
* Extracts damping-related parameters
* Produces comparative plots for each energy level

This workflow is representative of tasks in materials characterisation, ion-beam diagnostics and applied computational physics.

## Skills Demonstrated

* Scientific programming in Python
* Data parsing and cleaning
* Curve fitting with SciPy
* Numerical modelling of physical behaviour
* Visualisation using Matplotlib
* Reproducible analysis of experimental datasets

## Running the Script

Install dependencies:

```
pip install -r requirements.txt
```

Run the analysis:

```
python3 magnetic_flux_density_analysis.py
```

Plots and outputs will appear automatically.

## Dataset Description

The file `experimental_signal_data.dat` contains:

* A metadata header
* A ten-column data table consisting of:

  * Time (microseconds) and detector channel readings
  * For each of five implantation energies (5, 10, 15, 20, 25 keV)

The script skips metadata rows and automatically extracts the appropriate numerical data.

## Outputs

The script produces:

* Magnetic flux density as a function of time
* Damping behaviour visualisations
* Comparative plots between different implantation energies

These results illustrate how magnetic response varies with input energy.

## License

This project is licensed under the MIT License.
