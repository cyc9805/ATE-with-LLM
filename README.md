# Automatic Term Extraction with LLMs

## Introduction
This repository contains the code and resources for the paper [**Enhancing Automatic Term Extraction with Large Language Models via Syntactic Retrieval**](https://aclanthology.org/2025.findings-acl.516.pdf)

```
@inproceedings{chun-etal-2025-enhancing,
    title = "Enhancing Automatic Term Extraction with Large Language Models via Syntactic Retrieval",
    author = "Chun, Yongchan  and
      Kim, Minhyuk  and
      Kim, Dongjun  and
      Park, Chanjun  and
      Lim, Heuiseok",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.516/",
    pages = "9916--9926",
    ISBN = "979-8-89176-256-5"
}
```

## Requirements
Install the required Python packages listed in `requirements.txt` using:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
We use three datasets for our experiments: **ACTER**, **ACL-RD**, and **BCGM**. Follow these steps to prepare the datasets:

1. Navigate to the `src/dataset` directory and execute all cells in the `preprocess.ipynb` notebook located in each dataset folder.
2. Run the following command to create dataset indices for retrieval:
    ```bash
    python src/dataset/create_index.py
    ```

## Running Experiments
### Reproducing Main Results
To reproduce the main results from the paper:
1. Navigate to the `src/run_scripts` directory:
    ```bash
    cd src/run_scripts
    ```
2. Execute the script:
    ```bash
    bash test.sh
    ```

Alternatively, you can manually run the tests using:
```bash
cd src
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
  cd src/run_scripts
  bash train.sh
  bash test.sh
  ```

## Results Analysis
1. Experiment results are saved in the `src/outputs` directory.
2. To reproduce **Table 2** and **Figure 2** from the paper, execute all cells in the `src/analysis.ipynb` notebook.

