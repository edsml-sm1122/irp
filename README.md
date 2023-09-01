# Surrogate Modeling for Hybrid Wave Downscaling using MLP and LSTM

(in Python)

## Introduction

This repository houses the codebase and additional resources for the research conducted on utilizing Multilayer Perceptrons (MLPs) and Long Short-Term Memory networks (LSTMs) as surrogate models in the domain of hybrid wave downscaling. The objective is to predict Significant Wave Height (SWH) at coastlines from wave-related parameters at offshore 25m contour points. The research investigates the model's performance and generalization abilities across different spatial and temporal scales.

The research location is shown [here](https://github.com/ese-msc-2022/irp-sm1122/blob/main/codeFinal/datapre/map.html) The models predict from wave related variables at red points (25m contour) to significant wave height at blue points(coastline). 

---

## Table of Contents

1. [Documentation](#documentation)
2. [Code Structure](#code-structure)
3. [Getting Started](#getting-started)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)

---

## Documentation

Detailed documentation for each aspect of the work, including the architecture of the MLP and LSTM models, data preprocessing steps, and analysis strategy, can be found [here](https://github.com/ese-msc-2022/irp-sm1122/blob/main/reports/sm1122-finalreport.pdf). The readme (this) is a brief introduction to help make sense the codes. 

A brief documentaion for part of the functions can be found [here](https://github.com/ese-msc-2022/irp-sm1122/tree/main/codeFinal/doc/build/html). 

Please downlad the files and open any .html.

---

## Code explain

The codes are in folder codeFinal. This is just part of the work more relative to the report. For more work, please see the [Google Drive](https://drive.google.com/drive/folders/1l__vH-LfxBH7RtKKLTlmnXzO2cO7QeSU?usp=drive_link).

Most of the work developed on Google colab.  

```diff

The data is  <span style="color:red">*CONFEDENTIAL*</span>.

```
Some of the models are uploaded [here](https://github.com/ese-msc-2022/irp-sm1122/tree/main/codeFinal/models). However, this is not the file structure when models are developed, you won't be able to run the codes without changing path.

### Data Reading, Vusulization, Analysis
- `datapre`:
  - [link to codes](https://github.com/ese-msc-2022/irp-sm1122/tree/main/codeFinal/datapre)
  - `read_data.ipynb` data reading code from a local dataset with quite complex structure.
  - `explore_data.ipynb` check duplication, nan; plots of data, correlation analysis for better understanding 
  - `pointsLoc.ipynb` a code mainly to download and save elavation data for model target location from Google Map.
    
### Surragate models
- `Building_models`: Core implementation of the surrogate models.
  - [Link to Codes](https://github.com/ese-msc-2022/irp-sm1122/tree/main/codeFinal/Building_models)
  - `LSTM.ipynb` Develop the lstm model
  - `MLP.ipynb` Develop the MLP model

### Model generalization and performance
- `ModelTest_and_Geberalization`: Generalization of model configurations. Several new models are trained and tested.
  - [Link to Codes](https://github.com/ese-msc-2022/irp-sm1122/tree/main/codeFinal/ModelTest_and_Generalization)
  - `space_generalization`  as its name. Note, the functions and classes in these codes are not all merged into `utilities.py` as they are context-specific. 
  - `General_performance.ipynb` performace of the two models developed in `Building_models` 
  - `MLP_unseentrack.ipynb` train an MLP on 80% of storm event, test on the left
  - `MLP_LSTM_timeunseen.ipynb` train a lstm and mlp on the initial 80% of the time series, test on the remaining

### utilities and test
  - [`utilities.py`](https://github.com/ese-msc-2022/irp-sm1122/blob/main/codeFinal/utilities.py): utilities for codes developed pn Google Colab (codes that train/run models).
  - [`datafunctions.py`](https://github.com/ese-msc-2022/irp-sm1122/blob/main/codeFinal/datapre/data_functions.py) utilities for codes in the datapre folder.
  - [`test.ipynb`](https://github.com/ese-msc-2022/irp-sm1122/blob/main/codeFinal/tests.ipynb) testing functions in the two sforementioned `.py` files. The file is not written in test.py, but in .ipynb considering inconvenience in importing local packages and run .py on a Google Colab Virtual Machine.
    
---

## Getting Started

To clone this repository and set up a local development environment, follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/projectname.git`
2. Navigate to the project directory: `cd projectname`
3. Install dependencies: `pip install -r requirements.txt`

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/ese-msc-2022/irp-sm1122/edit/main/codeFinal/LICENSE.md) file for details.

---

## Acknowledgments

- This work has been conducted as an external msc project between Imperial College London and Moody's RMS.
- Supervisors: Dr. Christopher Thomas, Prof. Matthew Piggott

