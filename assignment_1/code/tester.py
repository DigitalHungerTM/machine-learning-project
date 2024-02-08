import pickle

def main():
    ### look at pickled data
    with open("data/vocab.pickle", "rb") as pickle_file:
        vocab = pickle.load(pickle_file)
        for ngram in vocab[0:-1]:
            print(ngram)


if __name__ == "__main__":
    main()
