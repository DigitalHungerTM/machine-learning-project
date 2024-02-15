import pandas as pd
import matplotlib.pyplot as plt


def read_dataset(filename: str):
    """
    read csv file and return it as pandas dataframe.
    it takes the first line of the file as column names
    :param `filename`: name of the file (excluding extension)
    :return `inputdf`: pandas df of the data
    """
    with open(f'{filename}', encoding='utf-8') as infile:
        inputdf = pd.read_csv(infile, sep=",", encoding='utf-8', header=0)

    return inputdf


def main():
    df = read_dataset("reporting/n_fold_new.csv")
    metric = 'euclidean'
    # for n in range(1,5):
    #     new_df = df[((df.distance_metric == metric) & (df.n == n))][['k', 'avg_macro_f1']]
    #     plt.plot(new_df.k, new_df.avg_macro_f1, label=f'{n=}')
    # plt.title(f'{metric.title()} n-fold validation')
    # plt.xlabel('k')
    # plt.ylabel('avg macro f1')
    # plt.legend()
    # plt.pause(0.01)
    # plt.savefig(f'reporting/{metric}_n_fold.png')
    # plt.show()

    # metrics = ['euclidean', 'cosine']
    # for metric in metrics:
    #     new_df = df[((df.distance_metric == metric) & (df.n == 4))][['k', 'avg_macro_f1']]
    #     plt.plot(new_df.k, new_df.avg_macro_f1, label=f'{metric}')

    # plt.title(f'Best n for both metrics')
    # plt.xlabel('k')
    # plt.ylabel('avg macro f1')
    # plt.legend()
    # plt.pause(0.01)
    # plt.savefig(f'reporting/best_n_comparison.png')
    # plt.show()

    df_token = read_dataset("reporting/n_fold_tokens.csv")
    df_char = read_dataset("reporting/n_fold_new.csv")
    # for n in range(1, 5):
    #     new_df = df_token[((df_token.distance_metric == metric) & (df_token.n == n))][['k', 'avg_macro_f1']]
    #     plt.plot(new_df.k, new_df.avg_macro_f1, label=f'{n=}')

    for metric in ['cosine']:
        new_df_token = df_token[((df_token.distance_metric == metric) & (df_token.n == 4))]
        plt.plot(new_df_token.k, new_df_token.avg_macro_f1, label=f'word based')
    for metric in ['cosine']:
        new_df_char = df_char[((df_char.distance_metric == metric) & (df_char.n == 4))]
        plt.plot(new_df_char.k, new_df_char.avg_macro_f1, label=f'character based')

    # plt.legend()
    # plt.title(f'Token vs character, best n')
    # plt.xlabel('k')
    # plt.ylabel('avg macro f1')
    # plt.savefig(f'reporting/token_character_comparison.png')
    # plt.show()

    # metric = 'cosine'
    # k = 5
    # new_df_token = df_token[((df_token.distance_metric == metric) & (df_token.k == k))]
    # plt.plot(new_df_token.n, new_df_token.avg_macro_f1, label=f'word based')
    # new_df_char = df_char[((df_char.distance_metric == metric) & (df_char.k == k))]
    # plt.plot(new_df_char.n, new_df_char.avg_macro_f1, label=f'character based')

    plt.legend()
    plt.title(f'word vs character')
    plt.xlabel('k')
    plt.ylabel('avg macro f1')
    plt.savefig(f'reporting/word_vs_character_n.png')
    plt.pause(0.01)
    plt.show()


if __name__ == "__main__":
    main()
