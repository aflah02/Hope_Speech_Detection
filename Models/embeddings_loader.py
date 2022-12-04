import numpy as np
import os
import pickle

def load_custom_embeddings(path = None, gensim = True):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        custom_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        custom_path = path

    aug_emb = None
    train_emb = None
    dev_emb = None
    test_emb = None
    if gensim:
        with open(os.path.join(custom_path, 'gensim_custom_w2v_embeddings_augmented_train.pickle'), 'rb') as handle:
            aug_emb = pickle.load(handle)
        with open(os.path.join(custom_path, 'gensim_custom_w2v_embeddings_train.pickle'), 'rb') as handle:
            train_emb = pickle.load(handle)
        with open(os.path.join(custom_path, 'gensim_custom_w2v_embeddings_dev.pickle'), 'rb') as handle:
            dev_emb = pickle.load(handle)
        with open(os.path.join(custom_path, 'gensim_custom_w2v_embeddings_test.pickle'), 'rb') as handle:
            test_emb = pickle.load(handle)
    else:
        with open(os.path.join(custom_path, 'custom_w2v_augmented_train.pickle'), 'rb') as handle:
            aug_emb = pickle.load(handle)
        with open(os.path.join(custom_path, 'custom_w2v_train.pickle'), 'rb') as handle:
            train_emb = pickle.load(handle)
        with open(os.path.join(custom_path, 'custom_w2v_dev.pickle'), 'rb') as handle:
            dev_emb = pickle.load(handle)
        with open(os.path.join(custom_path, 'custom_w2v_test.pickle'), 'rb') as handle:
            test_emb = pickle.load(handle)
    return aug_emb, train_emb, dev_emb, test_emb

def load_glove_twitter_25(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        gt25_save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        gt25_save_path = path
    train = np.load(os.path.join(gt25_save_path, 'gt_25_train_english.npy'), allow_pickle=True)
    dev = np.load(os.path.join(gt25_save_path, 'gt_25_dev_english.npy'), allow_pickle=True)
    test = np.load(os.path.join(gt25_save_path, 'gt_25_test_english.npy'), allow_pickle=True)
    return train, dev, test

def load_fasttext_300(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        ft300_save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        ft300_save_path = path
    train = np.load(os.path.join(ft300_save_path, 'ft_300_train_english.npy'), allow_pickle=True)
    dev = np.load(os.path.join(ft300_save_path, 'ft_300_dev_english.npy'), allow_pickle=True)
    test = np.load(os.path.join(ft300_save_path, 'ft_300_test_english.npy'), allow_pickle=True)
    return train, dev, test

def load_word2vec_300(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        wv300_save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        wv300_save_path = path
    train = np.load(os.path.join(wv300_save_path, 'w2v_300_train_english.npy'), allow_pickle=True)
    dev = np.load(os.path.join(wv300_save_path, 'w2v_300_dev_english.npy'), allow_pickle=True)
    test = np.load(os.path.join(wv300_save_path, 'w2v_300_test_english.npy'), allow_pickle=True)
    return train, dev, test

def load_labels(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        labels_save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        labels_save_path = path
    train = np.load(os.path.join(labels_save_path, 'train_labels.npy'), allow_pickle=True)
    dev = np.load(os.path.join(labels_save_path, 'dev_labels.npy'), allow_pickle=True)
    test = np.load(os.path.join(labels_save_path, 'test_labels.npy'), allow_pickle=True)
    return train, dev, test

def load_augmented_labels(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        labels_save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        labels_save_path = path
    train = np.load(os.path.join(labels_save_path, 'augmented_train_labels.npy'), allow_pickle=True)
    dev = np.load(os.path.join(labels_save_path, 'dev_labels.npy'), allow_pickle=True)
    test = np.load(os.path.join(labels_save_path, 'test_labels.npy'), allow_pickle=True)
    return train, dev, test

def load_tfidf_no_pca(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        tfidf_save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        tfidf_save_path = path
    train = np.load(os.path.join(tfidf_save_path, 'train_english_tfidf.npy'), allow_pickle=True)
    dev = np.load(os.path.join(tfidf_save_path, 'dev_english_tfidf.npy'), allow_pickle=True)
    test = np.load(os.path.join(tfidf_save_path, 'test_english_tfidf.npy'), allow_pickle=True)
    return train, dev, test

def load_tfidf_pca(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        tfidf_save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        tfidf_save_path = path
    train = np.load(os.path.join(tfidf_save_path, 'train_english_tfidf_pca.npy'), allow_pickle=True)
    dev = np.load(os.path.join(tfidf_save_path, 'dev_english_tfidf_pca.npy'), allow_pickle=True)
    test = np.load(os.path.join(tfidf_save_path, 'test_english_tfidf_pca.npy'), allow_pickle=True)
    return train, dev, test

def load_sent_trans_better_no_pca(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        save_path = path
    x = np.load(os.path.join(save_path, 'english_train_folds_sentence_model1_prePCA.npy'), allow_pickle=True)
    train, dev, test = x[0]
    return train, dev, test

def load_sent_trans_better_pca(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        save_path = path
    x = np.load(os.path.join(save_path, 'english_train_folds_sentence_model1_postPCA.npy'), allow_pickle=True)
    train, dev, test = x[0]
    return train, dev, test

def load_sent_trans_fast_no_pca(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        save_path = path
    x = np.load(os.path.join(save_path, 'english_train_folds_sentence_model2_prePCA.npy'), allow_pickle=True)
    train, dev, test = x[0]
    return train, dev, test

def load_sent_trans_fast_pca(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        save_path = path
    x = np.load(os.path.join(save_path, 'english_train_folds_sentence_model2_postPCA.npy'), allow_pickle=True)
    train, dev, test = x[0]
    return train, dev, test

def load_sent_trans_augmented_no_pca(path = None):
    if path == None:
        parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        save_path = os.path.join(parent_dir, 'Word_Embeddings\Pre Computed Word Embeddings')
    else:
        save_path = path
    x = np.load(os.path.join(save_path, 'augmented_english_train_folds_sentence_model1_prePCA.npy'), allow_pickle=True)
    train, dev, test = x[0]
    return train, dev, test

def load_only_2_class(embed, aug=False):
    if aug:
        train_labels, dev_labels, test_labels = load_augmented_labels()
    else:
        train_labels, dev_labels, test_labels = load_labels()
    if embed == 'tfidf':
        train, dev, test = load_tfidf_no_pca()
    elif embed == 'tfidf_pca':
        train, dev, test = load_tfidf_pca()
    elif embed == "w2v":
        train, dev, test = load_word2vec_300()
    elif embed == "fasttext":
        train, dev, test = load_fasttext_300()
    elif embed == "glove":
        train, dev, test = load_glove_twitter_25()
    elif embed == "sent_trans_better_no_pca":
        train, dev, test = load_sent_trans_better_no_pca()
    elif embed == "sent_trans_better_pca":
        train, dev, test = load_sent_trans_better_pca()
    elif embed == "sent_trans_fast_no_pca":
        train, dev, test = load_sent_trans_fast_no_pca()
    elif embed == "sent_trans_fast_pca":
        train, dev, test = load_sent_trans_fast_pca()
    elif embed == "sent_trans_augmented_no_pca":
        train, dev, test = load_sent_trans_augmented_no_pca()
    
    # If label is not-English drop the sample
    train = train[train_labels != 2]
    dev = dev[dev_labels != 2]
    test = test[test_labels != 2]

    # Remove The Labels
    train_labels = train_labels[train_labels != 2]
    dev_labels = dev_labels[dev_labels != 2]
    test_labels = test_labels[test_labels != 2]

    return train, dev, test, train_labels, dev_labels, test_labels