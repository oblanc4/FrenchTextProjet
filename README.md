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
- [Contributing](#contributing)
- [License](#license)

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
Provide information about the embeddings used in this project, their purpose, and how they were created.

## Post Training
Explain any processes or analysis conducted after training the model.

## Streamlit
Describe how Streamlit is implemented in the project, including instructions for running the Streamlit application.

## Installation
Instructions on installing and setting up your project.

## Usage
Guidelines on how to use the project, with examples of commands or scripts.

## Contributing
Information for those who wish to contribute to the project.

## License
The license under which this project is released.
