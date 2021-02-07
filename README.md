# LIT - **L**abeling **It**eratively

...is a simple measure to minimize the manual labeling effort.
`LIT` is a helper for finding the sample most valuable samples to label in a continuous training set-up.

## How it works

Starting with a few labeled data, a classifier needs to be trained. Given this MVP-classifier, the remaining unlabeled data
is classified. Those samples where the classifier performs worst i.e. the classifier cannot safely assign it one particular
class are selected for further manual labeling. The number of newly manual labeled samples can be varied based on the
actual classification task.

The main two assumptions in order to use `LIT` are:

* A basic model trained with as much labeled data as needed for a decent performance is available.

* A huge amount of unlabeled data is available.  

From there on, `LIT` will work as follows:

1) Run the trained model on the labeled data,
2) Identify the samples that are classified with highest uncertainty.
3) `LIT` returns `x` samples (as specified in the configuration) to be labeled manually.
4) `x` samples need to be labeled manually.
5) `LIT` trains the classifier again with all available labeled data samples.

Steps 1) to 5) are iteratively executed until a satisfying classification performance is reached.

## Background

The approach was analyzed in the course of a masters thesis on semantic radar grid maps for a binary classification task.
The active labeling approach was the most promising one to reduce manual effort. For more detailled information, the corresponding paper can be found at
[https://ieeexplore.ieee.org/abstract/document/8008123](https://ieeexplore.ieee.org/abstract/document/8008123).
