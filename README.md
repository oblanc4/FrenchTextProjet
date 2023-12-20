# 📚 Detecting the Difficulty Level of French Texts

We have embarked on the development of models aimed at predicting the difficulty level of a written text in French for English speakers. These models are designed to assess the level of linguistic proficiency required to comprehend and engage with the text, using a scale ranging from A1 to C2. This initiative aligns with our commitment to facilitate the learning of French for English speakers by providing automated assessment of text difficulty levels, thereby contributing to personalized learning experiences and making language acquisition more accessible and effective.

## 📑 Table of Contents
- [🏆 Results](#-results)
- [📊 Dataset and Dataset Upgrade](#-dataset-and-dataset-upgrade)
- [🧠 Embeddings](#-embeddings)
- [🔍 Post Training](#-post-training)
- [🌐 Streamlit](#-streamlit)
- [🎥 Demonstration Video](#-demonstration-video)

## 🏆 Results
The performance metrics below were calculated using a train-test split to evaluate the models. For the SVM model, the reported accuracy was obtained by applying the fine-tuned model to a new, unlabeled dataset.

| Metric       | Logistic Regression | kNN    | Decision Tree | Random Forests | 👑 SVM Fine-Tuning+Features |
|--------------|---------------------|--------|---------------|----------------|-----------------|
| Precision    | 0.474               | 0.348  | 0.324         | 0.410          | -               |
| Recall       | 0.475               | 0.232  | 0.326         | 0.416          | -               |
| F1-score     | 0.469               | 0.181  | 0.315         | 0.390          | -               |
| Accuracy     | 0.475               | 0.232  | 0.326         | 0.416          | 0.607           |

- Note: The SVM model underwent a fine-tuning process, significantly improving its metrics, with an overall accuracy of 0.914 on a new dataset.

## 📊 Dataset and Dataset Upgrade
In this project, we utilize a variety of data files to train and test our models. The process involves both initial dataset preparation and subsequent upgrades to enhance the data quality and relevance.

### Original Dataset
- `training_data.csv`: The training set, containing labeled data for model training.
- `unlabelled_test_data.csv`: The test set, used for model evaluation, consisting of unlabelled data.

### Dataset Enhancement
Using the `data_upgrade.ipynb` notebook, we've augmented the original datasets with additional attributes to improve the model's performance. This includes adding word count, POS tagging, and complexity metrics. The enhanced datasets are:
- `training_dataUP.csv`: Enhanced version of the original `training_data.csv`.
- `unlabelled_test_dataUP.csv`: Upgraded version of the `unlabelled_test_data.csv`.
- `augmented_training_dataUP.csv`: An augmented version of `training_data.csv`, which includes new sentences and is further enhanced using `data_upgrade.ipynb`.

## 🧠 Embeddings
This project explores various embedding methods to enhance its model's performance. The following notebooks document the tests conducted with different embedding techniques:
- `CamemBERT+Features.ipynb`: Testing embedding with CamemBERT, combined with additional features.
- `FlauBERT+Features.ipynb`: Experimentation with FlauBERT embeddings and additional feature integration.
- `NeuralNetworks.ipynb`: Trials with embedding techniques using neural networks.
- `RoBERTa+Features.ipynb`: Investigating the use of RoBERTa embeddings alongside additional features.
- `Sentence-CamemBERT-Large+Features.ipynb`: Exploring larger CamemBERT models at the sentence level with feature enhancement.


## 🔍 Post Training
After the initial training phase, we conducted a series of fine-tuning and evaluation processes to optimize our models further. The following notebooks and files are involved in this post-training phase:

- `SVM Fine-Tuning+feature.ipynb`: This notebook contains our best-performing model with fine-tuning applied, alongside additional feature integration.
- `SVM-Fine-Tuning.ipynb`: This notebook details the fine-tuning process for the SVM model without additional features.
- `model_only.pth`: The PyTorch model file that contains the trained model weights. You can download this file from [this link](https://www.swisstransfer.com/d/9ae9ec06-7742-4cb3-9f2d-005a7f800af6) (This link expires on 18.01.2024 at 22:07).

These materials provide an in-depth look at the refinement steps taken to enhance model performance and achieve the reported results.


## 🌐 Streamlit
To use the Streamlit application:
1. Download the `modele_camembert` folder from [this link](https://www.swisstransfer.com/d/31832bd3-57c7-4c0e-a43b-5bccc74879a5) and add it to the `streamlit` directory in your project (This link expires on 18.01.2024 at 19:56).
2. Modify the file paths in `Myapp.py` to use relative paths:
    - Tokenizer path: `./streamlit/tokenizer`
    - Model path: `./streamlit/modele_camembert`
    - SVM model path: `./streamlit/svm_model.pkl`
3. To run the application, open a terminal and execute: streamlit run [relative/path/to/Myapp.py]. Make sure to replace `[relative/path/to/Myapp.py]` with the actual relative path to `Myapp.py`.

## 🎥 Demonstration Video
Watch the following video for a detailed explanation of the most effective model, SVM Fine-Tuning+Features, and to see the Streamlit application in action:

[![SVM Fine-Tuning and Streamlit Application](http://img.youtube.com/vi/INsprDhmOUA/0.jpg)](https://youtu.be/INsprDhmOUA)

Click the image above to play the video.
