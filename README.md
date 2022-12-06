# Federated Learning for SMS Spam Detection

## Dependencies
Install any necessary dependencies by running the first cell, and the cell right under it imports the necessary packages. 

The imblearn package is open-source and provdes code to implement many classic balancing techniques. We used the imblearn package to implement all balancing techniques except for Augmented Synonyms, Augmented Context Substitute, and Augmented Context Insert. 

We also made use of a couple of NLP Data Augmentation techniques, such as Augmented Synonyms, Augmented Context Substitute, and Augmented Context Insert. To implement these in our code, we used the open source nlpaug library, which provides this functionality. 

We used the Flower Library to simulate Federated Learning, and our first two cells include the code to install and import the Flower Library.

## Dataset
We obtained our SMS data from kaggle, which can be viewed [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). Since our code uses pre-trained transformers available on HuggingFace, this dataset was uploaded to a private HuggingFace repository. 

## Balancing Dataset
The cells under the Balancing Dataset Section take the original SMS dataset and apply a number of class balancing techniques on it. The balancing techniques we implemented are [Naive Random Oversampling](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html), [Naive Random Undersampling](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html), [CNN Undersampling](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.CondensedNearestNeighbour.html), [Synonym Oversampling](https://github.com/makcedward/nlpaug), [Context Substitute Oversampling](https://github.com/makcedward/nlpaug), [Context Insert Oversampling](https://github.com/makcedward/nlpaug), and [Oversampled SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html). We have each technique implemented in its own cell in that order and the implementation details are explained in the comments in their respective cells. 

## Centralized Model Configuration and Training
The Centralized Model section in our code creates and trains the centralized model we will be using as a baseline to compare to our federated models. Our centralized model uses BERT Tiny, which is avaiable on HuggingFace and is already pre-trained. We also have code cells that evaluate our centralized model and generate confusion matrices for both the test dataset and a few real world spam messages we pulled from our phones.
Since SMOTE doesn't work with the transformer model we are using, we used a Passive Aggressive classifier instead of BERT Tiny. We applied all the steps we used for the Transformer Based centralized model on the Passive Aggressive classifier we used with SMOTE.

## Federated Learning Model Configuration and Training
Our Federated Learning experiments were done using 5 clients with varying distributions of legitimate and spam messages on each client. Our first cell in the federated learning section equally splits the training data across the 5 clients we are using. It splits the data across the clients such that each client will have the same number of data points with a varying distribution of spam. We used four spam distributions in our experiment: a completely random class distribution which was obtained from randomly sampling 20% of the dataset for each client (this was experiment -1 in our code), [50%,50%,50%,50%,50%], [0%,0%,50%,100%,100%], [15%,30%,50%,70%,85%]. We repeated this experiment across all datasets, including the original unbalanced dataset and the datasets obtained from each of balancing techniques. 

## Note
All code should be run sequentially to guarantee results

## Link to our demo video - https://youtu.be/vbGhMupOF9o
