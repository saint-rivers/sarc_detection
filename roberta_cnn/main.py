import gc
from copy import deepcopy as dc

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from tokenizers.implementations import ByteLevelBPETokenizer
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
import math
from copy import deepcopy as dc
import gc
import model as mymodel

import manipulate

print('TF version', tf.__version__)

MAX_LEN = 96
PATH = './'
tokenizer = ByteLevelBPETokenizer(
    vocab=PATH + 'vocab-roberta-base.json',
    merges=PATH + 'merges-roberta-base.txt',
    lowercase=True,
    add_prefix_space=True
)
EPOCHS = 1  # originally 3
BATCH_SIZE = 32  # originally 32
PAD_ID = 1
SEED = 88888
LABEL_SMOOTHING = 0.1
tf.random.set_seed(SEED)
np.random.seed(SEED)
sentiment_id = {'positive': 1313, 'neutral': 7974, 'negative': 2430}

train = pd.read_csv('train.csv').fillna('')
# if you directly want the result database, just uncomment the following line :
# train = pd.read_csv('../input/extended-train-for-tweet/extended_train.csv')
# train.dropna(inplace=True)
print(train.head())

print("Downloading Natural Language Toolkit (NLTK) libraries")
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

print("testing synonyms")
for word in ['sad', 'SAD', 'Sad...', 'saaaaad']:
    print(f'For word {word}, synonyms and corresponding probabilities are :')
    print(manipulate.get_synonyms(word))
    print('-' * 20)

# for _ in range(5):
#     print(manipulate.swap_words('The sun is shining today, this makes me feel so '
#                              'good !')[0])

manipulate.new_row(train.loc[np.random.choice(train.shape[0])], n_samples=8)

temp = [manipulate.new_row(row, n_samples=2) for _, row in train.iterrows()]
augmented_data = pd.concat(temp, axis=0)#.sample(frac=1)
train['number'] = [t.shape[0] for t in temp]
train['number'] = train['number'].cumsum()
del temp
gc.collect()
augmented_data.drop_duplicates(subset=['text'], inplace=False, ignore_index=True)
augmented_data.reset_index(drop=True, inplace=True)
# augmented_data.head(20)
match_index = dc(train['number'])
train.drop(columns='number', inplace=True)
match_index = [0] + match_index.values.tolist()
match_borders = list(zip(match_index[:-1], match_index[1:]))
del match_index
gc.collect()
train['brackets'] = match_borders
#
#
# augmented_data.to_csv('extended_train.csv', index=False)
# train = augmented_data
# del augmented_data

# train = pd.read_csv("extended_train.csv")
train['text_len'] = train['text'].apply(len)
train.hist(column='text_len')
train.loc[train.text_len<150].hist(column='text_len')
train.loc[train.text_len>=150, 'textID'].apply(lambda x: 'new' in x).describe()

# select only less than 150 characters, for efficiency in training
train = train.loc[train.text_len<150]
train.drop(columns=["text_len"], inplace=True)
train.reset_index(drop=True, inplace=True)
train.to_csv('extended_train.csv', index=False)

# augmented_data = pd.read_csv("extended_train.csv").fillna('')
# train = pd.read_csv("extended_train.csv").fillna('')

####################
##### tokenize #####
####################

#1
print("##### started tokenizer #####")

ct = augmented_data.shape[0]
input_ids = np.ones((ct, MAX_LEN), dtype='int32')
attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
start_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
end_tokens = np.zeros((ct, MAX_LEN), dtype='int32')

for k, row in augmented_data.iterrows():

    # FIND OVERLAP
    text1 = " " + " ".join(row['text'].split())
    text2 = " ".join(row['selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx + len(text2)] = 1
    if text1[idx - 1] == ' ':
        chars[idx - 1] = 1
    enc = tokenizer.encode(text1)

    # ID_OFFSETS
    offsets = []
    idx = 0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx, idx + len(w)))
        idx += len(w)

    # START END TOKENS
    toks = []
    for i, (a, b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm > 0:
            toks.append(i)

    s_tok = sentiment_id[row['sentiment']]
    input_ids[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]
    attention_mask[k, :len(enc.ids) + 3] = 1
    if len(toks) > 0:
        start_tokens[k, toks[0] + 2] = 1
        end_tokens[k, toks[-1] + 2] = 1

#2

ct_train = train.shape[0]
input_ids_train = np.ones((ct_train, MAX_LEN), dtype='int32')
attention_mask_train = np.zeros((ct_train, MAX_LEN), dtype='int32')
token_type_ids_train = np.zeros((ct_train, MAX_LEN), dtype='int32')
start_tokens_train = np.zeros((ct_train, MAX_LEN), dtype='int32')
end_tokens_train = np.zeros((ct_train, MAX_LEN), dtype='int32')

for k, row in train.iterrows():

    # FIND OVERLAP
    text1 = " " + " ".join(row['text'].split())
    text2 = " ".join(row['selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx + len(text2)] = 1
    if text1[idx - 1] == ' ':
        chars[idx - 1] = 1
    enc = tokenizer.encode(text1)

    # ID_OFFSETS
    offsets = []
    idx = 0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx, idx + len(w)))
        idx += len(w)

    # START END TOKENS
    toks = []
    for i, (a, b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm > 0:
            toks.append(i)

    s_tok = sentiment_id[row['sentiment']]
    input_ids_train[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]
    attention_mask_train[k, :len(enc.ids) + 3] = 1
    if len(toks) > 0:
        start_tokens_train[k, toks[0] + 2] = 1
        end_tokens_train[k, toks[-1] + 2] = 1

#3
test = pd.read_csv('./test.csv').fillna('')

temp = [manipulate.test_new_row(row, n_samples=2) for _, row in
        test.iterrows()]
test_augmented_data = pd.concat(temp, axis=0)#.sample(frac=1)
test['number'] = [t.shape[0] for t in temp]
test['number'] = test['number'].cumsum()
del temp
gc.collect()
test_augmented_data.drop_duplicates(subset=['text'], inplace=False, ignore_index=True)
test_augmented_data.reset_index(drop=True, inplace=True)
test_match_index = dc(test['number'])
test.drop(columns='number', inplace=True)
test_match_index = [0] + test_match_index.values.tolist()
test_match_borders = list(zip(test_match_index[:-1], test_match_index[1:]))
del test_match_index
gc.collect()
test['brackets'] = test_match_borders

print(test_augmented_data.head(10))

ct = test.shape[0]
test_input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')

for k, row in test.iterrows():
    # INPUT_IDS
    text1 = " " + " ".join(row['text'].split())
    enc = tokenizer.encode(text1)
    s_tok = sentiment_id[row['sentiment']]
    test_input_ids_t[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]

ct = test_augmented_data.shape[0]
input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')

for k, row in test_augmented_data.iterrows():
    # INPUT_IDS
    text1 = " " + " ".join(row['text'].split())
    enc = tokenizer.encode(text1)
    s_tok = sentiment_id[row['sentiment']]
    input_ids_t[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]
    attention_mask_t[k, :len(enc.ids) + 3] = 1



##### training #####

jac = []
VER = 'v0'
DISPLAY = 1  # USE display=1 FOR INTERACTIVE
oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
oof_end = np.zeros((input_ids.shape[0], MAX_LEN))
preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)  #
# originally 5 splits
for fold, (idx_T, idx_V) in enumerate(skf.split(input_ids_train, train.sentiment.values)):
    idxT = np.array([i for (a, b) in train.loc[idx_T, 'brackets'] for i in range(a, b)])
    idxV = np.array([i for (a, b) in train.loc[idx_V, 'brackets'] for i in range(a, b)])
    print('#' * 25)
    print('### FOLD %i' % (fold + 1))
    print('#' * 25)

    K.clear_session()
    model, padded_model = mymodel.build_model()

    # sv = tf.keras.callbacks.ModelCheckpoint(
    #    '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
    #    save_weights_only=True, mode='auto', save_freq='epoch')
    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]
    targetT = [start_tokens[idxT,], end_tokens[idxT,]]
    inpV = [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]]
    targetV = [start_tokens[idxV,], end_tokens[idxV,]]
    # sort the validation data
    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))
    inpV = [arr[shuffleV] for arr in inpV]
    targetV = [arr[shuffleV] for arr in targetV]
    weight_fn = '%s-roberta-%i.h5' % (VER, fold)
    for epoch in range(1, EPOCHS + 1):
        # sort and shuffle: We add random numbers to not have the same order in each epoch
        shuffleT = np.int32(
            sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3),
                   reverse=True))
        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch
        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)
        batch_inds = np.random.permutation(num_batches)
        shuffleT_ = []
        for batch_ind in batch_inds:
            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])
        shuffleT = np.concatenate(shuffleT_)
        # reorder the input data
        inpT = [arr[shuffleT] for arr in inpT]
        targetT = [arr[shuffleT] for arr in targetT]
        model.fit(inpT, targetT,
                  epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY, callbacks=[],
                  validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`
        mymodel.save_weights(model, weight_fn)

    print('Loading model...')
    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    mymodel.load_weights(model, weight_fn)

    print('Predicting OOF...')
    oof_start[idxV,], oof_end[idxV,] = padded_model.predict(
        [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=DISPLAY)

    print('Predicting Test...')
    preds = padded_model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
    preds_start += preds[0] / skf.n_splits
    preds_end += preds[1] / skf.n_splits

    # DISPLAY FOLD JACCARD
    alls = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])
        if a > b:
            st = augmented_data.loc[k, 'text']  # IMPROVE CV/LB with better choice here
        else:
            text1 = " " + " ".join(augmented_data.loc[k, 'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a - 2:b - 1])
        alls.append(mymodel.jaccard(st, augmented_data.loc[k,
        'selected_text']))
    jac.append(np.mean(alls))
    print('>>>> FOLD %i Jaccard =' % (fold + 1), np.mean(alls))
    print()

print('>>>> OVERALL 3Fold CV Jaccard =',np.mean(jac))
print(jac)