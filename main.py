import pandas as pd


def main():


    from data.data_loader import load_data, print_data_summary
    from training.data_split import stratified_split, verify_split_stratification

    # Load the data
    data = load_data('data/intent_data.csv')

    # Print summary
    print_data_summary(data)

    # Split the data
    train_data, val_data, test_data = stratified_split(
        data, 
        train_ratio=0.8, 
        test_ratio=0.1, 
        random_seed=42
    )

    # Verify stratification
    verify_split_stratification(train_data, val_data, test_data)


if __name__ == "__main__":
    main()