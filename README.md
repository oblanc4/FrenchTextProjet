# Detecting the Difficulty Level of French Texts

We have embarked on the development of models aimed at predicting the difficulty level of a written text in French for English speakers. These models are designed to assess the level of linguistic proficiency required to comprehend and engage with the text, using a scale ranging from A1 to C2. This initiative aligns with our commitment to facilitate the learning of French for English speakers by providing automated assessment of text difficulty levels, thereby contributing to personalized learning experiences and making language acquisition more accessible and effective.

## Table of Contents
- [Dataset](#dataset)
- [Dataset Upgrade](#dataset-upgrade)
- [Embeddings](#embeddings)
- [Post Training](#post-training)
- [Streamlit](#streamlit)
- [Installation](#installation)
- [Usage](#usage)

## Dataset
This section of the project uses several key data files:
- `training_data.csv`: The training set, containing labeled data for model training.
- `unlabelled_test_data.csv`: The test set, used for model evaluation, consisting of unlabelled data.

## Dataset Upgrade
In this phase, the dataset is enhanced using the `data_upgrade.ipynb` notebook, which adds additional attributes like word count, POS tagging, and complexity to the dataframes. The upgraded datasets include:
- `training_dataUP.csv`: The original `training_data.csv` enhanced with additional attributes.
- `unlabelled_test_dataUP.csv`: Similar to `training_dataUP`, this is the upgraded version of `unlabelled_test_data.csv`.
- `augmented_training_dataUP.csv`: This file takes the original `training_data.csv`, augments it with new sentences, and then enhances it using `data_upgrade.ipynb`.


## Embeddings
This project explores various embedding methods to enhance its model's performance. The following notebooks document the tests conducted with different embedding techniques:
- `CamemBERT+Features.ipynb`: Testing embedding with CamemBERT, combined with additional features.
- `FlauBERT+Features.ipynb`: Experimentation with FlauBERT embeddings and additional feature integration.
- `NeuralNetworks.ipynb`: Trials with embedding techniques using neural networks.
- `RoBERTa+Features.ipynb`: Investigating the use of RoBERTa embeddings alongside additional features.
- `Sentence-CamemBERT-Large+Features.ipynb`: Exploring larger CamemBERT models at the sentence level with feature enhancement.


## Post Training
Explain any processes or analysis conducted after training the model.

## Streamlit
To use the Streamlit application:
1. Download the `modele_camembert` folder from [this link](https://www.swisstransfer.com/d/31832bd3-57c7-4c0e-a43b-5bccc74879a5) and add it to the `streamlit` directory in your project (This link expires on 18.01.2024 at 19:56).
2. Modify the file paths in `Myapp.py` to use relative paths:
    - Tokenizer path: `./streamlit/tokenizer`
    - Model path: `./streamlit/modele_camembert`
    - SVM model path: `./streamlit/svm_model.pkl`
3. To run the application, open a terminal and execute: streamlit run [relative/path/to/Myapp.py]. Make sure to replace `[relative/path/to/Myapp.py]` with the actual relative path to `Myapp.py`.


## Installation
Instructions on installing and setting up your project.

## Usage
Guidelines on how to use the project, with examples of commands or scripts.
