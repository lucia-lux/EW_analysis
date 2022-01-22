
import re
import pkg_resources
import nltk
import numpy as np
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from symspellpy import SymSpell, Verbosity
from nltk.stem.porter import PorterStemmer

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def func_norm(s):
    """
    Perform some basic normalisation operations.

    Parameters
    ----------
    s:  str
        text to operate on
    
    Returns
    -------
        Normalised string
    
    """
    s = s.lower() # lower case
    # letter repetition (>2)
    s  = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # non word repetition
    s = s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # noise text
    s = re.sub(r' ing ', ' ', s)
    # phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
    return s.strip()

def func_punc(w_list):
    """
    Remove non-alphabet characters. Includes punctuation.

    Parameters
    ----------
    w_list: list
        list of tokens to be processed
    
    Returns
    -------
        list without non-alphabet characters
    """
    return [word for word in w_list if word.isalpha()]

def func_stopf(w_list):
    """
    Remove stop words

    Parameters
    ----------
    w_list: list
        list of tokens to be processed
    
    Returns
    -------
        list without stop words
    """
    stop_words = set(stopwords.words('english'))
    w_list  = [f for f in w_list if f not in stop_words]
    return w_list


def func_stem(w_list):
    """
    stem word list

    Parameters
    ----------
    w_list: list
        word list for stemming

    Returns
    -------
        stemmed word list
    """
    pstem = PorterStemmer()
    sw_list = [pstem.stem(w) for w in w_list]
    return sw_list

def func_noun(w_list):
    """
    Select nouns only from w_list.
    
    Parameters
    ----------
    w_list: list[str]
        word list to be processed
    
    Returns
    -------
    list[str] of nouns in w_list
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']

def func_verb(w_list):
    """ 
    Parameters
    ----------
    w_list: list[str]
        list of tokens to be processed
    Returns
    -------
        list of verbs only
    """
    return [word for (word,pos) in nltk.pos_tag(w_list) if pos[:2] == 'VB']

def func_adjective(w_list):
    """ 
    Parameters
    ----------
    w_list: list
        list of tokens to be processed
    Returns
    -------
        list of verbs only
    """
    return [word for (word,pos) in nltk.pos_tag(w_list) if pos[:2] == 'JJ']

def func_inf_words(w_list):
    """ 
    Retain verbs, adjectives and nouns only

    Parameters
    ----------
    w_list: list
        list of words to be processed
    Returns
    -------
    list of nouns, adjectives and verbs only
    """
    return [word for (word,pos) in nltk.pos_tag(w_list) if 'VB' in pos or 'JJ' in pos or 'NN' in pos]


def func_spell(w_list):
    """
    in: word list to be processed
    out: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        if any(map(word.__contains__, ['covid','lockdown'])):
            w_list_fixed.append(word)
        else:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
            if suggestions:
                w_list_fixed.append(suggestions[0].term)
            else:
                pass
    return w_list_fixed

def get_pos_tag(tag):
    """
    convert NLTK pos tag to wordnet
    pos tag format.

    Parameters
    ----------
    tag: str
        NLTK pos tag
    
    Returns
    -------
        wordnet pos tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def func_lemmatize(word_list):
    """
    Lemmatize word list.

    Parameters
    ----------
    word_list:  list
        words to process
    
    Returns
    -------
    Lemmatized word list.

    """
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.pos_tag(word_list)
    words_lemmatized = [lemmatizer.lemmatize(word,get_pos_tag(tag))
                        for (word,tag) in word_list]
    return words_lemmatized

def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw texts

    Parameters
    ----------
    rw: str
            sentence to be processed

    Returns
    -------
        sentence level pre-processed text
    """
    s = func_norm(rw)

    return s



def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences.
    
    Parameters
    ----------
    s:  str
        sentence to be processed
    
    Returns
    -------
        word level pre-processed text
    """
    if not s:
        return None
    w_list = word_tokenize(s)
    w_list = func_punc(w_list)
   # w_list = func_inf_words(w_list)
    w_list = func_spell(w_list)
    w_list = func_lemmatize(w_list)
    w_list = func_stopf(w_list)

    return w_list

def preprocess(docs):
    """
    Preprocess the data.

    Parameters
    ----------
    docs: list
        list of documents to be preprocessed
    
    Returns
    -------
        Preprocessed sentences, tokens
    """
    print('Preprocessing raw texts ...')
    #n_docs = len(docs)
    sentences = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed
    #samp = np.random.choice(n_docs)
    for i in range(0, len(docs)):
        sentence = preprocess_sent(docs.iloc[i])
        token_list = preprocess_word(sentence)
        if token_list:
            sentences.append(sentence)
            token_lists.append(token_list)
        print('{} %'.format(str(np.round((i + 1) / len(docs) * 100, 2))), end='\r')
    print('Preprocessing raw texts. Done!')
    return sentences, token_lists