from datasets import load_dataset
import pandas as pd
def get_dataset(language):
    dataset = load_dataset("hope_edi", language)
    train = dataset["train"].to_pandas()
    validation = dataset["validation"].to_pandas()
    return train, validation

if __name__ == "__main__":
    languages = ["english", "tamil", "malayalam"]
    for language in languages:
        train, validation = get_dataset(language)
        train.to_csv(f"Data/{language}_train.csv", index=False)
        validation.to_csv(f"Data/{language}_validation.csv", index=False)