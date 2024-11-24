# Turkish Delights: A Dataset on Turkish Euphemisms
Euphemisms are a form of figurative language often used to soften sensitive topics or avoid direct language. 
Despite their importance in communication, they remain relatively understudied in natural language processing (NLP). 
This repository introduces the Turkish Potentially Euphemistic Terms (PETs) Dataset, the first publicly available resource of its kind in the field.

This repository contains the following:

- Creating a curated list of 64 euphemistic terms (PETs) commonly used in Turkish.
- Collecting contextual examples from real-world sources.
- Annotating these examples to distinguish euphemistic and non-euphemistic uses of PETs.

## Overview
Euphemisms are often used to soften sensitive topics or avoid harsh language. This project focuses on 64 Potentially Euphemistic Terms (PETs) in Turkish, providing valuable resources for tasks like euphemism detection and cross-linguistic analysis.

This dataset includes 6,115 labeled examples, covering both euphemistic and non-euphemistic instances. The dataset focuses on 64 Potentially Euphemistic Terms (PETs) across various domains of sensitive language. Each instance is labeled and annotated with [PET_BOUNDARY] markers for easy identification of euphemistic phrases.

## Repository Contents
### Datasets
1. **`turkish_pets_full_dataset.csv`**
- Contains 6,115 labeled examples of euphemistic and non-euphemistic instances.
- Includes 64 PETs categorized into 10 categories (e.g., death, employment, politics).

**Columns:**
- **num:** ID, the number of example
- **PET:** Potentially Euphemistic Term.
- **variation:** The morphological variation of the PET.
- **category:** Category of the PET.
- **orig_text, clean_text:** Original and lowercased text.
- **char_count, word_count:** Character and word counts.
- **edited_text:** Text with [PET_BOUNDARY] markers for PETs.
- **label:** Binary annotation (1 = euphemistic, 0 = non-euphemistic)

2. **`turkish_pets_balanced_dataset.csv`**
- **Size**: 908 examples (521 euphemistic, 387 non-euphemistic).
- Simplified for classification tasks using transformer-based models.

3. **`Turkish_PETs_List.pdf`**
- A supplementary resource listing additional euphemistic terms that were not represented in the datasets due to a lack of examples in the corpus.

## How to Use

This section explains how to set up and run experiments using the Turkish Potentially Euphemistic Terms (PETs) dataset. Follow these steps to reproduce the results or adapt the setup for your own experiments.

---

### **Step 1: Clone the Repository**
Clone this repository to your local machine:
```bash
git clone --branch experiment https://github.com/hasancanbiyik/Turkish_PETs.git
cd Turkish_PETs
```

### Step 2: Create a virtual environment (optional)
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r experiment/EXPERIMENT_RUNNER/requirements.txt
```

### Step 4: Enable permissions to execute the script.

```bash
chmod +x local/run.sh
```

### Step 5: Run the experiment

Run the run.sh script

```bash
# Execute the shell script for a predefined experiment
bash local/run.sh
```

### Step 6: Evaluating the model
Run the launch.py script to evaluate the fine-tuned model:

```bash
python3 src/launch.py --split_path splits/fold_0/test.csv --model_path experiment_xlmr/saved_models/model_checkpoint.bin
```

## Cross-Validation Splits
The splits/ folder contains **20 folds** of cross-validation splits for fine-tuning and evaluating models.

## Fine-Tuned Models
Fine-tuned models for **XLM-R**, **mBERT**, **BERTurk**, and **ELECTRA** will be hosted on **[Hugging Face](https://huggingface.co/hasancanbiyik/)**. 
These models will allow researchers to benchmark their approaches without requiring additional fine-tuning.

Stay tuned for updates!

## Citation
If you use this dataset or repository in your research, please cite our paper:

- Hasan Biyik, Patrick Lee, and Anna Feldman. 2024. Turkish Delights: a Dataset on Turkish Euphemisms. In Proceedings of the First Workshop on Natural Language Processing for Turkic Languages (SIGTURK 2024), pages 71–80, Bangkok, Thailand and Online. Association for Computational Linguistics.

## Contact
For questions or suggestions, feel free to reach out:

**Hasan Can Biyik** 

Research Assistant, Montclair State University

[Email](biyikh1@montclair.edu) / biyikh1@montclair.edu

