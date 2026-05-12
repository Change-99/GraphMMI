# miRNA_transfer Readme

This repository contains the code and data for reproducing the results presented in the accompanying paper. The repository is structured as follows:

- **notebook**: This directory contains the Jupyter notebooks with the scripts used to generate the results in the paper.
- **src**: The source code used by the notebook scripts is stored in this directory.
- **data**: The data directory includes the datasets in various preprocessing stages, ranging from raw data to processed data.
- **models**: All trained models, as well as their output files such as prediction files and visualization images, are saved in this directory.

To run the code and reproduce the results, follow these steps:

1. Install the required dependencies by creating a Conda environment using the `environment.yml` file. Run the following command: `conda env create -f environment.yml`.
2. Add the raw datasets into the `data/raw` folder.
3. Run the `preprocessing.py` script in the `notebook` directory. This script will preprocess the raw data into the desired format.
4. Finally, run the `overall_run.py` script in the `notebook` directory to execute the overall analysis and generate the results.

By following these steps, you will be able to reproduce the results presented in the paper using the provided code and datasets.
