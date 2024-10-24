# from tokenizers import ByteLevelBPETokenizer
from copy import deepcopy as dc
from nltk.corpus import wordnet, stopwords

import numpy as np
import pandas as pd
from nltk.corpus import wordnet
import string

punc = list(string.punctuation)
punc.remove("-")
punc.append(" ")
stop = stopwords.words('english')
stop += ["_________________________________", "u"]


def get_synonyms(word):
    """
    Get synonyms of a word
    """
    if word.lower() in stop:
        return [word], [1]

    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word not in synonyms:
        synonyms.add(word)

    n = len(synonyms)

    if n == 1:  # we didn't find any synonyms for that word, therefore we will try to check if it's not because of some punctuation interfering
        word_ = "".join(list(filter(lambda x: x not in punc, word)))
        if word_.lower() in stop:
            return [word, word_], [0.5, 0.5]
        for syn in wordnet.synsets(word_):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word_ not in synonyms:
            synonyms.add(word_)

    n = len(synonyms)
    if n == 1:
        probabilities = [1]
    else:
        probabilities = [0.5 if w == word else 0.5 / (n - 1) for w in synonyms]

    return list(synonyms), probabilities


def swap_words(words):
    words = words.split()
    if len(words) < 2:
        return " ".join(words), False
    random_idx = np.random.randint(0, len(words) - 1)
    words[random_idx], words[random_idx + 1] = words[random_idx + 1], words[random_idx]
    return " ".join(words), True


def new_row(row, n_samples=1):
    text, selected_text, textID = row['text'], row['selected_text'], row['textID']
    print(text)
    oth_text = text.replace(selected_text, " _________________________________ ")
    new_selected_text = [get_synonyms(word) for word in selected_text.split()]
    new_oth_text = [get_synonyms(word) for word in oth_text.split()]
    new_sentences = [row]
    for i in range(n_samples):
        oth_text_ = " ".join([np.random.choice(l_syn, p=p, replace=True) for l_syn, p in new_oth_text])
        selected_text_ = " ".join([np.random.choice(l_syn, p=p, replace=True) for l_syn, p in new_selected_text])
        text_ = oth_text_.replace("_________________________________", selected_text_)
        if not selected_text_ in text_:
            print(
                f'Original : {text} with target {selected_text}, oth_text {oth_text}\nTransformed : {text_} with target {selected_text_}, oth_text {oth_text_}')
            continue
        row2 = dc(row)
        row2['text'] = text_
        row2['selected_text'] = selected_text_
        row2['textID'] = f'new_{textID}'
        new_sentences.append(row2)
    for r in dc(new_sentences):
        r_ = dc(r)
        if np.random.choice([True, False]):
            selected_text, boo = swap_words(r_['selected_text'])
            if boo:
                r_['text'] = r_['text'].replace(r_['selected_text'], selected_text)
                r_['selected_text'] = selected_text
            else:
                oth_text, _ = swap_words(r_['text'].replace(r_['selected_text'], " _________________________________ "))
                r_['text'] = oth_text.replace("_________________________________", r_['selected_text'])
        else:
            oth_text, _ = swap_words(r_['text'].replace(r_['selected_text'], " _________________________________ "))
            r_['text'] = oth_text.replace("_________________________________", r_['selected_text'])
        r_['textID'] = f'new_{textID}'
        new_sentences.append(r_)
    new_rows = pd.concat(new_sentences, axis=1).transpose().drop_duplicates(subset=['text'], inplace=False,
                                                                            ignore_index=True)
    new_rows = new_rows.loc[new_rows['text'].apply(len) < 150]
    counter = 0
    for i, row in new_rows.iterrows():
        if row['textID'][:4] == 'new_':
            row['textID'] = row['textID'] + f'_{counter}'
            counter += 1
    return new_rows


def test_new_row(row, n_samples=1):
    text, textID = row['text'], row['textID']
    new_text = [get_synonyms(word) for word in text.split()]
    new_sentences = [row]
    for i in range(n_samples):
        text_ = " ".join([np.random.choice(l_syn, p=p, replace=True) for l_syn, p in new_text])
        row2 = dc(row)
        row2['text'] = text_
        row2['textID'] = f'new_{textID}'
        new_sentences.append(row2)
    for r in dc(new_sentences):
        r_ = dc(r)
        text, boo = swap_words(r_['text'])
        if boo:
            r_['text'] = text
            r_['textID'] = f'new_{textID}'
            new_sentences.append(r_)
    new_rows = pd.concat(new_sentences, axis=1).transpose().drop_duplicates(subset=['text'], inplace=False,
                                                                            ignore_index=True)
    new_rows = new_rows.loc[new_rows['text'].apply(len) < 150]
    counter = 0
    for i, row in new_rows.iterrows():
        if row['textID'][:4] == 'new_':
            row['textID'] = row['textID'] + f'_{counter}'
            counter += 1
    return new_rows
