#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 21:40:34 2021

@author: nickfang
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import os as os
import glob
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

os.chdir('/Users/nickfang/Desktop/IMT 574/a6')
directory = os.getcwd()


youtube = pd.read_csv('Youtube01-Psy.csv')
katy = pd.read_csv('Youtube02-KatyPerry.csv')
lmfao = pd.read_csv('Youtube03-LMFAO.csv')
eminem = pd.read_csv('Youtube04-Eminem.csv')
shakira = pd.read_csv('Youtube05-Shakira.csv')

train_df = pd.DataFrame()
#Read data
files = glob.glob("./YouTube-Spam-Collection-v1/*")
files = sorted(files)[:-1]
for f in files:
    temp = pd.read_csv(f)
    print("(file, shape):", f, temp.shape)
train_df = train_df.append(temp, ignore_index=True)
print("the shape of training files", train_df.shape)

#Pick necessary columns
train_df = train_df[['CONTENT', 'CLASS']]
#train_df['CONTENT'] = train_df['CONTENT'].apply(textcleaner)
train_df.head()

test_df = pd.read_csv('./YouTube-Spam-Collection-v1/Youtube05-Shakira.csv')
print("the shape of the test dataset", test_df.shape)
test_df = test_df[['CONTENT', 'CLASS']]

from sklearn.feature_extraction.text import CountVectorizer

classifier = Pipeline([
    ('bow', CountVectorizer()), # strings to token integer counts
    ('classifier', MultinomialNB()),
    ])

#Now all you have to do is find your class prediction on test content.

train_df['CLASS'].value_counts(normalize=True)
test_df['CLASS'].value_counts(normalize=True)

train_df.head(3)


train_df['CONTENT'] = train_df['CONTENT'].str.replace(
    '\W', ' ')
train_df['CONTENT'] = train_df['CONTENT'].str.lower()
train_df.head(3)

train_df['CONTENT'] = train_df['CONTENT'].str.split()

vocabulary = []
for sms in train_df['CONTENT']:
   for word in sms:
      vocabulary.append(word)

vocabulary = list(set(vocabulary))
len(vocabulary)

word_counts_per_sms = {unique_word: [0] * len(train_df['CONTENT']) for unique_word in vocabulary}
for index, content in enumerate(train_df['CONTENT']):
   for word in content:
      word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()

training_set_clean = pd.concat([train_df, word_counts], axis=1)
training_set_clean.head()

# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['CLASS'] == 1]
messages = training_set_clean[training_set_clean['CLASS'] == 0]

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_message = len(messages) / len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['CONTENT'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_message = messages['CONTENT'].apply(len)
n_messages = n_words_per_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1

parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_messages = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_messages = messages[word].sum() # ham_messages already defined
   p_word_given_messages = (n_word_given_messages + alpha) / (n_messages + alpha*n_vocabulary)
   parameters_messages[word] = p_word_given_messages
   
import re

def classify_test_set(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_word_given_messages = p_message

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_messages:
         p_word_given_messages *= parameters_messages[word]

   if p_word_given_messages > p_spam_given_message:
      return 0
   elif p_spam_given_message > p_word_given_messages:
      return 1
   else:
      return 2

test_df['predicted'] = test_df['CONTENT'].apply(classify_test_set)
test_df.head()

correct = 0
total = test_df.shape[0]

for row in test_df.iterrows():
   row = row[1]
   if row['CLASS'] == row['predicted']:
      correct += 1

print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)



def classify(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_word_given_messages = p_message

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_messages: 
         p_word_given_messages *= parameters_messages[word]

   print('P(Spam|message):', p_spam_given_message)
   print('P(Ham|message):', p_word_given_messages)

   if p_word_given_messages > p_spam_given_message:
      print('Label: Real')
   elif p_word_given_messages < p_spam_given_message:
      print('Label: Spam')
   else:
      print('Equal proabilities')





















