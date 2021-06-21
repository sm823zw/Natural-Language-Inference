# Natural-Language-Inference

We are given a pair of sentences, “premise” and “hypothesis”. Given that the “premise” is true, the task is to determine whether the “hypothesis” is true(entailment), false(contradiction), or undetermined(neutral). The Stanford Natural Language Interface (SNLI) corpus has been used. It is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI).

The official homepage for this corpus is [here](https://nlp.stanford.edu/projects/snli/). Following are some of the samples illustrating the three classes of output we expect the model to predict.

Example 1 –> 

Premise – A soccer game with multiple males playing. 

Hypothesis – Some men are playing a sport.

Output – Entailment

Example 2 –>

Premise – A black race car starts up in front of a crowd of people. 

Hypothesis – A man is driving down a lonely road.

Output – Contradiction

Example 3 –>

Premise –  An older and younger man smiling. 

Hypothesis –Two men are smiling and laughing at the cats playing on the floor.

Output – Neutral

In this project, several Bi-LSTM and Bi-GRU models were created and trained for the task of NLI. We implemented the baseline model introduced by [Bownman et. al.](https://nlp.stanford.edu/pubs/snli_paper.pdf) who published the dataset. To improve model performances, several attention mechanisms such as [1-way attention](https://arxiv.org/pdf/1509.06664.pdf), [2-way attention](https://arxiv.org/pdf/1509.06664.pdf), and [Inner Attention](https://arxiv.org/pdf/1605.09090.pdf) have been implemented. Several [sentence matching techniques](https://arxiv.org/pdf/1512.08422.pdf) have been applied to enhance the model performance.

The project has been implemented in Python (3.8.5) programming language. Tensorflow (2.4) framework was used for building and training the models. spacy, contractions, word2number, and BeatifulSoup libraries were used for textual data pre-processing. Seaborn library was used to plot the confusion matrix.
