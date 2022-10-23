import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re, emoji, os

def replace_username(post: str) -> str:
    post = re.sub('@[^\s]+', '<user>', post)
    return post

def replace_urls(post: str) -> str:
    post = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", "<url>", post)
    post = re.sub(r'http.*?(?=\s)', "<url>", post)
    post = re.sub(r"http\S+", "<url>", post)
    return post

def remove_whitespaces(post: str) -> str:
    post = ' '.join(post.split())
    return post

def remove_stopwords(post: list[str]) -> list[str]:
    stopwords = stopwords.words('english')
    altered_text = [w for w in post if w not in stopwords]
    return altered_text

def tokenize(text: str) -> list[str]:
    tk = TweetTokenizer()
    tokenized = tk.tokenize(text)
    return tokenized

def remove_punctuation(post: str) -> str:
    post = re.sub('([*]+)|([/]+)|([\]+)|([\)]+)|([\(]+)|([:]+)|([#]+)|([\.]+)|([,]+)|([-]+)|([!]+)|([\?])|([;]+)|[\']|[\"]', '', post)
    return post

def remove_emoji(post: str) -> str:
    altered_text = emoji.replace_emoji(post, replace='<emoji>')
    return altered_text

def preprocess(text: str, stopwords_remove: bool = False):
    text = text.lower()
    text = replace_urls(text)
    text = replace_username(text)
    text = remove_punctuation(text)
    text = remove_whitespaces(text)
    text = remove_emoji(text)
    text = tokenize(text)
    if stopwords_remove:
        text = remove_stopwords(text)
    return ' '.join(text)

def split_csv(df):
    df.columns = ["test"]
    df["text"] = df.apply(lambda row: ' '.join(row["test"].split(";")[:-2]), axis = 1)
    df["label"] = df.apply(lambda row: row["test"].split(";")[-2], axis = 1)
    df = df.drop(columns=["test"])
    return df

if __name__ == "__main__":
    df_english_train = pd.read_csv("Data/OldData/english_train.csv", header = None)
    df_english_dev = pd.read_csv("Data/OldData/english_dev.csv", header = None)
    df_english_test = pd.read_csv("Data/OldData/english_test.csv", header = None)
    df_english_train = split_csv(df_english_train)
    df_english_dev = split_csv(df_english_dev)
    df_english_test = split_csv(df_english_test)
    
    df_english_train["preprocessed_text"] = df_english_train.apply(lambda row: preprocess(row.text), axis = 1)
    df_english_dev["preprocessed_text"] = df_english_dev.apply(lambda row: preprocess(row.text), axis = 1)
    df_english_test["preprocessed_text"] = df_english_test.apply(lambda row: preprocess(row.text), axis = 1)

    df_english_train.to_csv("Data/PreprocessedData/english_train_preprocess.csv")
    df_english_dev.to_csv("Data/PreprocessedData/english_dev_preprocess.csv")
    df_english_test.to_csv("Data/PreprocessedData/english_test_preprocess.csv")