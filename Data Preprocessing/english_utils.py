
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import emoji

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

def remove_punctuation(post : str) -> str:
    post = re.sub('([*]+)|([/]+)|([\]+)|([\)]+)|([\(]+)|([:]+)|([#]+)|([\.]+)|([,]+)|([-]+)|([!]+)|([\?])|([;]+)|[\']|[\"]', '', post)
    return post

def remove_emoji(post : str) -> str:
    altered_text = emoji.replace_emoji(post, replace='')
    return altered_text

if __name__ == "__main__":
    df = pd.read_csv("../Data/english_train.csv")