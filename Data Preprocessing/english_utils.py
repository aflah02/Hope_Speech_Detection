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
    return text

if __name__ == "__main__":
    df_english = pd.read_csv(os.path.dirname(__file__) + "/../Data/english_train.csv")
    df_english["tokenized_text"] = df_english.apply(lambda row: preprocess(row["text"]), axis = 1)
    df_english.to_csv(os.path.dirname(__file__) + "/PreprocessedData/english_train_preprocessed.csv")