import pandas as pd


def main():
    df = pd.read_csv("./data/intent_data.csv")
    print(df.head())
    print(df.info())
    print(df.labels.value_counts())




if __name__ == "__main__":
    main()