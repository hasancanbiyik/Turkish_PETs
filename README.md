# Turkish Delights: A Dataset on Turkish Euphemisms

This repository contains files accompanying the SIGTURK 2024 paper, Turkish Delights: A Dataset on Turkish Euphemisms, presented at SIGTURK 2024 (co-located with ACL 2024). The dataset is designed to support research on euphemistic language use in Turkish, with a focus on understanding and identifying euphemistic expressions across contexts.

## Overview

Euphemisms are often used to soften sensitive topics or avoid harsh language. This dataset focuses on Turkish euphemisms, providing a valuable resource for research in Natural Language Processing (NLP), Computational Linguistics, and Cultural Studies.

### Key Features:
Balanced Dataset: Contains 908 examples labeled as euphemistic or non-euphemistic.
Marked Expressions: Includes [PET_BOUNDARY] tags to highlight euphemistic phrases in context.
Multilingual Potential: Can be extended to comparative studies across languages.

## Dataset Details

The repository includes the following files:

**1. turkish_pets_list.csv**

A list of Potentially Euphemistic Terms (PETs) gathered from various sources, including Turkish media, social networks, and literature.
Columns: term, frequency, context

**2. tr_pets_balanced_dataset.csv**

A dataset of 908 sentences labeled as euphemistic (1) or non-euphemistic (0). Each euphemistic expression is marked with [PET_BOUNDARY] tags for easier identification.
Columns: text, label

**3. data_description.txt**

A file explaining the structure and purpose of the datasets.
