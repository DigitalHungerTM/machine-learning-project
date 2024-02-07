# Machine Learning Project Assignment 1 Report

Mathijs Afman - s4765079

## log

Everything went pretty smoothly up until the first test run of the provided SVM
classifier, where I found out the reason for storing the vocabulary of the training
set (the hard way, after fitting a model and attempting to predict on test data).

### Log of the first succesful run of the provided SVM classifier

```
importing...
importing done, processing...
***** Reading the dataset *****
Number of samples: 1950
***** Reading the dataset *****
Number of samples: 534

training set
vocabulary length:     24802
number of features:    24802

test set
number of features:    24802

preprocessing done, fitting...

fitting done, predicting...
predicting done, evaluating...
***** Evaluation *****

            accuracy:  0.612
            precision: [0.587 0.714]
            recall:    [0.894 0.298]
            f1:        [0.709 0.42 ]

            macro avg:
            precision: 0.651
            recall:    0.596
            f1:        0.565
        
done, took 512.7 seconds
```

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
