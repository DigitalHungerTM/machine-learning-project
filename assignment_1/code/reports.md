# Machine Learning Project Assignment 1 Report

Mathijs Afman - s4765079

## log

Everything went pretty smoothly up until the first test run of the provided SVM
classifier, where I found out the reason for storing the vocabulary of the training
set (the hard way, after fitting a model and attempting to predict on test data).

## feature extraction method

- Has support for n-grams.
- If generated (not passed as argument), will return a vocabulary
- vocab generator can be ran on its own

## preprocessor

removes the following things from tweets:

- user tags
- hastags
- websites
- punctuation
- emojis
- capitalization

## evaluator

evaluates

- accuracy
- precision
- recall
- f1-score

and macro averages of the latter three
