# Hope Speech Detection: Identifying Positive Actors in Toxic Discourses 

This repository contains our course project for [CSE-343 (Monsoon 2022)](http://techtree.iiitd.edu.in/viewDescription/filename?=CSE343).

*Accepted at ICLR 2023 for the Tiny Papers Track*

### What is Hope Speech?

Hope speech is any message or content that is positive, encouraging, reassuring, inclusive and supportive that inspires and engenders optimism in the minds of people.

### TLDR for our project?

We define two tasks:

Task 1: Multiclass Hope Speech Detection; In this task, we categorize the tweets into three classes, Hope, Non-Hope and Non-English.

Task 2: Two class classification; In this task we categorize the tweets as Hope and Non-Hope speech and drop the “Non-English” class. 

We match SOTA in Task 1 with simple ML Models while we beat SOTA in Task 2, and by a lot ;) using DL Models

# Dataset and Original Task

We use the dataset, made available by [Chakravarthi et al. 2022](https://aclanthology.org/2022.ltedi-1.58).
It consists of a very skewed distribution, with around 20k samples in favour of one class (`Non_Hope_Speech`) and only about 2k samples for the other class(es). The task was originally formulated as a three-way classification task, where we need to predict one of the labels `{Non_Hope_Speech, Hope_Speech, not-English}`.

# Methodology

- For Task 1, we tried out different word embedding techniques (`GloVe`, `FastText`, `word2vec`, `TF-IDF`, and `Sentence-BERT`) and also tried various combinations with them by performing PCA or leaving them as is, to see if we can retain some amount of data while also compressing the dimensions, which we have reported in our final results. We also tried to experiment with `custom word2vec` embeddings, which we created from scratch.
- We dumped the final embeddings for future use, and each of us then took up different types of Classifier models from sklearn, and performed the stated task using all the embeddings thus generated. Towards the end, we also tried different DL models like `BERT`, `BERTweet`, and other similar Pre-trained Transformer-based classifiers.
- We have reported the `Weighted F1 scores` for Task 1 as that is the metric which was used in the original paper. For Task 2 we take the top 5 ML models from Task 1 and also run LSTM, RNN and some pre-trained models for the 2 way classification and report the `Macro F1` scores.

# Results


Surpisingly enough, we managed to beat the SOTA results reported for both the tasks.
- For Task 1, we were able to do so using classical ML methods like `Linear Discriminant Analysis`.
- For Task 2, we beat SOTA here as well but this time we managed to do so by a large margin (by about 20 Macro F1 points) using DL methods.

### Our Top Models:
<img width="494" alt="image" src="https://user-images.githubusercontent.com/72096386/205753009-c4006adb-cb78-4ae7-afb1-62db5b8a241b.png">

Task 1 mimics the [First Shared Task](https://sites.google.com/view/lt-edi-2021/home) while Task 2 mimics the [Second Shared Task](https://sites.google.com/view/lt-edi-2022/home)

---

### Directory Structure:

- [`Data Preprocessing/`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/Data%20Preprocessing): Contains the Preprocessing Files
- [`Data`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/Data): Contains the Data
- [`DataAugmentation`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/DataAugmentation): Contains Data Augmentation Utils
- [`Word_Embeddings`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/Word_Embeddings): Contains the code to generate word embeddings and also the folder to store dump files. The word embeddings are hosted on Google Drive due to Space Constraints
- [`Documents`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/Documents): Our Reports, Presentations and Proposal
- [`Explainability`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/Explainability): Files to see evaluate models from an explainability lens
- [`Exploratory Data Analysis`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/Exploratory%20Data%20Analysis): Our Visualizations and Analysis
- [`Models`](https://github.com/aflah02/Hope_Speech_Detection/tree/main/Models): Contains the code for our models and also the folder to store the saves. The saves for all our ML models are hosted on Google Drive due to space constraints while our DL Model checkpoints are too large to shift and hence are not present on Google Drive but you can generate them by simply running the notebook

