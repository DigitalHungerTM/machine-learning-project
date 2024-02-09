# Machine Learning Project Assignment 1 Report

Mathijs Afman - s4765079

## log

Everything went pretty smoothly up until the first test run of the provided SVM
classifier, where I found out the reason for storing the vocabulary of the training
set (the hard way, after fitting a model and attempting to predict on test data).

### Log of the first succesful run of the provided SVM classifier

This was run with n-grams set to 2 tokens, on my windows machine. Which for some reason took 8.5 minutes.

```text
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

While working on my laptop (pop-os!, python3.10) I added pickling support for the vocab. I don't know if this is the reason but my laptop seems to be way faster for the fitter, even when generating 2-grams and 3-grams (which gicantically increases the vocab size, 6.500 > 30.000) it is done in a few seconds instead of literal minutes. Log with n=1 on my laptop:

```text
***** Reading the dataset *****

Number of samples: 1950

***** Reading the dataset *****

Number of samples: 534
***** Preprocessing *****

***** Preprocessing *****

training data
getting features
vocab pickle found, loading vocab pickle
changed n, regenerating vocab
vocab length:  6956
testing data
getting features
vocab pickle found, loading vocab pickle
***** Evaluating *****


            accuracy:  0.655
            precision: [0.648 0.667]
            recall:    [0.759 0.54 ]
            f1:        [0.699 0.596]

            macro avg:
            precision: 0.658
            recall:    0.649
            f1:        0.648
```

Log with n=3 on my laptop:

```text
***** Reading the dataset *****

Number of samples: 1950

***** Reading the dataset *****

Number of samples: 534
***** Preprocessing *****

***** Preprocessing *****

training data
getting features
vocab pickle found, loading vocab pickle
changed n, regenerating vocab
vocab length:  30376
testing data
getting features
vocab pickle found, loading vocab pickle
***** Evaluating *****


            accuracy:  0.551
            precision: [0.541 0.833]
            recall:    [0.989 0.06 ]
            f1:        [0.699 0.111]

            macro avg:
            precision: 0.687
            recall:    0.524
            f1:        0.405
```

I also noticed that I now had support for character based encoding instead of word based, so I added a unicode decoder that would convert every non unicode character (ë, à) to its unicode counterpart (e, a). Character based log with n=1:

```text
***** Reading the dataset *****

Number of samples: 534
training data

***** Preprocessing *****

***** Preprocessing *****

training data
***** Getting features *****

vocab pickle not found, generating vocab
vocab length:  39
testing data
***** Getting features *****

vocab pickle found, loading vocab pickle
***** Evaluating *****


            accuracy:  0.706
            precision: [0.696 0.721]
            recall:    [0.787 0.615]
            f1:        [0.739 0.664]

            macro avg:
            precision: 0.708
            recall:    0.701
            f1:        0.701
```

## feature extraction method

- Has support for n-grams
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
