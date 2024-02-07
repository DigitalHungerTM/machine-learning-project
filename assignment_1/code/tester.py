# timeit
from timeit import default_timer as timer
start = timer()
print("importing...")

# import functions that you want to test
from assignment_classification import *
from abs_custom_classifier_with_feature_generator import CustomClassifier
import numpy as np
import sys
from pprint import pprint
np.set_printoptions(threshold=sys.maxsize)
print("importing done, processing...")

def main():
    cc = CustomClassifier()
    texts, _ = read_dataset("data", "CT22_dutch_1B_claim_train")
    p_texts = preprocess_dataset(texts)
    features, vocab = cc.get_features(p_texts, n=5)
    print(f"vocabulary length:                                       {len(vocab)}")
    print(f"number of features (should be the same as vocab length): {len(features[0])}")
    stop = timer()
    print(f"processing done, took {stop - start:.1f} seconds")


if __name__ == "__main__":
    main()
