# codealong demo by https://www.humanitiesdataanalysis.org/vector-space-model/notebook.html




#import tokenizer
import nltk
import nltk.tokenize


#import regex
import re

#import collections counter
import collections

#import numpy for mapping
import numpy as np


# %%


# download the most recent punkt package
nltk.download('punkt', quiet = True)

# check if input string is either a single punctuation marker or a sequence thereof:
PUNCT_RE = re.compile(r'[^\w\s]+$')

def is_punct(string: str):
    """ Check if STRING is a punctuation marker or a sequence of punctuation markers
    Arguments:
        string (str): a string to check for punctuation markers
    Returns:
        bool: True if string is a sequence of punctuation markers, false otherwise
    Examples:
        >>> is_punct("!")
        True
        >>> is_punct("Bonjour!")
        False
        >>> is_punct("¿Te gusta el verano?")
        False
        >>> is_punct("...")
        True
    """
    return PUNCT_RE.match(string) is not None


def preprocess_text(text: str, lang: str, lowercase=True):
    """Preprocess text to tokens

    Args:
        text (str): a string representing a text
        lang (str): a string specifying the language of text
        lowercase (bool): Set to true to lowercase all word tokens. Defaults to True

    Returns:
        _type_: _description_
    """
    if lowercase:
        text = text.lower()
    
    tokens = nltk.tokenize.word_tokenize(text, language=lang)
    tokens = [token for token in tokens if not is_punct(token)]
    return tokens

def extract_vocabulary(tokenized_corpus: list, min_count=1, max_count=float('inf')):
    """Extract vocabulary from a tokenized corpus

    Args:
        tokenized_corpus (list): a tokenized corpus represented list of lists of strings
        min_count (int, optional): the minimum occurrence count of a vocabulary item in the corpus. Defaults to 1.
        max_count (_type_, optional): the maximum occurrence count of a vocabulary item in the corpus. Defaults to float('inf').
    """
    voc = collections.Counter()
    for doc in tokenized_corpus:
        voc.update(doc)
    
    voc = {word for word, count in voc.items()
            if count >= min_count and count <= max_count}
    
    return(sorted(voc))


def corpus2dtm(tokenized_corpus: list, vocabulary: list):
    """Transforms a tokenized corpus into a document-term matrix

    Args:
        tokenized_corpus (list): a tokenized corpus as a list of lists of strings
        vocabulary (list): a list of unique words

    Returns:
        _type_: a list of lists representing the frequency of each term in 'vocabulary' for each document in corpus
    """
    document_term_matrix = []

    for doc in tokenized_corpus:
        doc_count = collections.Counter(doc)
        row = [doc_count[word] for word in vocabulary]
        document_term_matrix.append(row)
    
    return document_term_matrix

corpus = [
    "D'où me vient ce désorde, Aufide, et que veut dire",
    "Madame, il était temps qu'il vous vint du secours:",
    "Ah! Moniseur, c'est donc vous?",
    "Ami, j'ai beau réver, toute ma rêverie",
    "Ne me parle plus tant de joie et d'hyménée;",
    "Il est vrai, Cléobule, et je veux l'avouer",
    "Laisse-moi mon chagrin, tout injuste qu'il est:",
    "Ton frère, je l'avoue, a beaucoup de mérite;",
    "J'en demeure d'accord, chacun a sa méthode;",
    "Pour prix de votre amour que vous peignez extrême,"
]

"""
#Tokenized
document3 = corpus[3]
print(nltk.tokenize.word_tokenize(document3, language='french'))

tokens = nltk.tokenize.word_tokenize(corpus[2], language='french')

tokenized = [token for token in tokens if not is_punct(token)]
print(tokenized)

#tokenize by preprocess function
for document in corpus[2:4]:
    print('Original:', document)
    print('Tokenized:', preprocess_text(document, 'french'))

#implement counter
vocabulary = collections.Counter()
for document in corpus:
    vocabulary.update(preprocess_text(document,'french'))

print('Most common words:')
print(vocabulary.most_common(n=5))

print('Original vocabulary size:', len(vocabulary))
pruned_vocabulary = {token for token, count in vocabulary.items() if count > 1}
print(pruned_vocabulary)
print('Pruned vocabulary size:', len(pruned_vocabulary))


tokenized_corpus = [preprocess_text(document, 'french') for document in corpus]
vocabulary = extract_vocabulary(tokenized_corpus)

bag_of_words = []
for document in tokenized_corpus:
    tokens = [word for word in document if word in vocabulary]
    bag_of_words.append(collections.Counter(tokens))
print(bag_of_words[2])


document_term_matrix = np.array(corpus2dtm(tokenized_corpus, vocabulary))
print(document_term_matrix.shape)
"""


# %%
#import matplotlib for graph

import matplotlib.pyplot as plt
import collections
#import components to download the play and manipulate tarball
import os
import lxml.etree
import tarfile

# open the source files
tf = tarfile.open('data/03-vector-space-model/data/theatre-classique.tar.gz', 'r')
tf.extractall('data')


subgenres = ('Comédie', 'Tragédie', 'Tragi-comédie')
plays, titles, genres =  [], [], []

for fn in os.scandir('data/03-vector-space-model/data/theatre-classique/'):
    
    # Only include XML files
    if not fn.name.endswith('.xml'):
        continue
    
    tree = lxml.etree.parse(fn.path)
    genre = tree.find('//genre')
    title = tree.find('//title')

    if genre is not None and genre.text in subgenres:
        lines = []
        for line in tree.xpath('//l|//p'):
            lines.append(' '.join(line.itertext()))
        
        text = '\n'.join(lines)
        plays.append(text)
        genres.append(genre.text)
        titles.append(title.text)

genre_counts = collections.Counter(genres)

fig, ax = plt.subplots()
ax.bar(genre_counts.keys(), genre_counts.values(), width = 0.3)
ax.set(xlabel = "Genre", ylabel = "Count")

plays_tok = [preprocess_text(play, 'french') for play in plays]
vocabulary = extract_vocabulary(plays_tok, min_count=2)
document_term_matrix = np.array(corpus2dtm(plays_tok, vocabulary))

print(f"document-term matrix with "
      f"|D| = {document_term_matrix.shape[0]} documents and "
      f"|V| = {document_term_matrix.shape[1]} words.")

# %%
moniseur_idx = vocabulary.index('moniseur')
print(moniseur_idx)
# %%
