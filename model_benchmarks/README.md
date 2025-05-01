# Benchmarks of the available AI models

This folder contains the benchmarks of the available multimodal embedding models and the benchmark of the available MLLMs. All Python scripts are made to be run from the folder they are located in!

## Folder structure (and how to run the benchmarks)

* **/embedding_benchmarks** - Folder containing the benchmarks of the multimodal embedding models
* **/mllm_benchmark** - Folder containing the benchmark of the MLLMs
* **requirments.txt** - The Python packages needed to run the benchmarks (but cetain models may require different packages or package versions

### Both folders contain:

* **/dataset** or **/datasets** - The data used for the benchmark(s)
* **/results** - Folder for storing the benchmark outputs and also the evaluated results (including plots)
* **run_benchmark.py** - Python script for running the benchmark with the selected model and outputting the non-evaluated results - the active model is configured inside this file, along with the active dataset (embedding benchmarks only)
* **data.py** - Contains functions manipulating with the datasets, also prints the dataset statistics when run directly
* **evaluate_results** - Reads the raw benchmark(s) results, evaluates them, and saves the evaluated results (including certain plots)
