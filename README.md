# Automatic Term Extraction with LLMs

## Requirements
Install the required Python packages listed in `requirements.txt` using:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
We use three datasets for our experiments: **ACTER**, **ACL-RD**, and **BCGM**. Follow these steps to prepare the datasets:

1. Navigate to the `dataset` directory and execute all cells in the `preprocess.ipynb` notebook located in each dataset folder.
2. Run the following command to create dataset indices for retrieval:
    ```bash
    python dataset/create_index.py
    ```

## Running Experiments
### Reproducing Main Results
To reproduce the main results from the paper:
1. Navigate to the `run_scripts` directory:
    ```bash
    cd run_scripts
    ```
2. Execute the script:
    ```bash
    bash test.sh
    ```

Alternatively, you can manually run the tests using:
```bash
python main.py
```
Key configuration arguments include:
- **config_path**: Path to the configuration file (e.g., `configs/test.json`).
- **model**: Model name (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`, `google/gemma-2-9b-it`, `mistralai/Mistral-Nemo-Instruct-2407`).
- **dataset**: Dataset name (`ACTER`, `ACL-RD`, `BCGM`).
- **num_shots**: Number of shots to use.
- **retrieval_method**: Retrieval method (`default`, `default_w_ins`, `bm25`, `random`, `fastkassim`).
- **seed**: Random seed (default: `42`).

### Comparison to Pretrained Language Models
To reproduce results from the "Comparison to Pretrained Language Models" section:
- For **RoBERTa**, execute all cells in the `RoBERTa.ipynb` notebook.
- For **BART**, run:
  ```bash
  cd run_scripts
  bash train.sh
  bash test.sh
  ```

## Results Analysis
1. Experiment results are saved in the `outputs` directory.
2. To reproduce **Table 2** and **Figure 2** from the paper, execute all cells in the `analysis.ipynb` notebook.

