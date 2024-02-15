import pickle
from assignment_classification import read_dataset

def main():
    _, train_labels = read_dataset("data", "CT22_dutch_1B_claim_train")
    n_labels = len(train_labels)
    n_1 = train_labels.count(1)
    n_0 = train_labels.count(0)
    p_1 = train_labels.count(1)/n_labels
    p_0 = 1-p_1
    print(f"""
{n_labels = }
{p_1 = :.2f}
{p_0 = :.2f}
{n_1 = }
{n_0 = }
""")


if __name__ == "__main__":
    main()
