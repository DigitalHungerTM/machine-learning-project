# timeit
from timeit import default_timer as timer
start = timer()
print("importing...")

# import functions that you want to test
from assignment_classification import read_dataset, preprocess_dataset, evaluate
from sklearn_svm_classifier import SVMClassifier
from abs_custom_classifier_with_feature_generator import CustomClassifier
import numpy as np
import sys
from pprint import pprint
np.set_printoptions(threshold=sys.maxsize) # dangerous when printing large datasets
print("importing done, processing...")

N = 2 # ngram setting

def main():
    ### feature getter ###
    cc = CustomClassifier()
    
    # training data
    train_texts, train_labels = read_dataset("data", "CT22_dutch_1B_claim_train")
    p_train_texts = preprocess_dataset(train_texts)
    # generate a vocabulary on the train data
    vocab = cc.gen_vocab(p_train_texts, n=N)
    # N-hot encode train texts on the vocabulary
    train_features = cc.get_features(p_train_texts, vocab=vocab, n=N)

    # testing data
    test_texts, true_test_labels = read_dataset("data", "CT22_dutch_1B_claim_dev_test")
    p_test_texts = preprocess_dataset(test_texts)
    # N-hot encode test texts on the vocabulary
    test_features = cc.get_features(p_test_texts, vocab=vocab, n=N)

    # print nicely
    print(
        f"""
training set
vocabulary length:     {len(vocab)}
number of features:    {len(train_features[0])}

test set
number of features:    {len(test_features[0])}

preprocessing done, fitting and predicting...
"""
    )

    ### SVM classifier fitter ###
    svm = SVMClassifier()
    svm.fit(train_features, train_labels)

    predicted_test_labels = svm.predict(test_features)

    print("fitting and predicting done, evaluating...")

    ### evaluator ###
    evaluate(true_test_labels, predicted_test_labels)

    # end of program
    stop = timer()
    print(f"processing done, took {stop - start:.1f} seconds")


if __name__ == "__main__":
    main()
