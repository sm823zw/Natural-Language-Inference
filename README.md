# Natural-Language-Inference

We are given a pair of sentences, “premise” and “hypothesis”. Given that the “premise” is true, the task is to determine whether the “hypothesis” is true(entailment), false(contradiction), or undetermined(neutral). The Stanford Natural Language Interface (SNLI) corpus, a collection of 570k human-written English sentence pairs manually labeled for classification with the labels entailment, contradiction, and neutral has been used.

The official homepage for this corpus is [here](https://nlp.stanford.edu/projects/snli/)

In this project, several Bi-LSTM and Bi-GRU models were created and trained for the task of NLI. To improve model performances, several attention mechanisms such as 1-way attention, 2-way attention, and Inner Attention have been implemented. Several sentence matching techniques have been applied to enhance the model performance.

The project has been implemented in Python programming language. Tensorflow framework was used for building and training the models. spacy, contractions, word2number, and BeatifulSoup libraries were used for textual data pre-processing. Seaborn library was used to plot the confusion matrix.
