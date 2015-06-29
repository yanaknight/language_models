#some NLP algorithms in Python (for language models)
import os
import re
import random
import math
from unidecode import unidecode
import numpy as np
from numpy.random import random_sample

#change file as required
file = 'training.de'

#read and lowercase
with open (file, "r") as myfile:
    data=myfile.read().lower().replace(" ", '')

#replace umlauts/accents
data1 = unidecode(data)

#replace anything non-alphabetic
data1 = re.sub("[^a-z]","",data1)

#def ngrams(data):
n=3

storage = dict()
storage1 = dict()

#create an n-gram list
for i in range(len(data1)-n+1):
  gram = tuple(data1[i:i+n])
  if gram in storage:
      storage[gram] += 1
  else:
      storage[gram] = 1

  if gram[:2] in storage1:
      storage1[gram[:2]] += 1
  else:
      storage1[gram[:2]] = 1

myfile.close()

#all poss trigrams
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#make a model
probabilities = dict()
probabilities_new = dict()
f = open('model_en', 'w')
leng = len(storage.items())
for key, value in storage.items():
  bigram = key[:2]
  prob = float(value)/float(storage1[bigram])
  probabilities[key] = prob

#all poss trigrams
for key, value in storage1.items():
  for i in letters:
      d = key + tuple(i)
      if d not in storage:
        storage[d] = 0

#smoothing
li = 0.0
for key, value in storage.items():
    bigram = key[:2]
    prob1 = float(value+1.0)/float(storage1[bigram]+26)
    probabilities_new[key] = prob1
    f.write(str(key) + ' ' + str(prob1) + '\n')
    if key[:2] == ('t', 'h'):# and key in probabilities:
      li+=probabilities_new[key]

print(li)

f.close()


#display all n-grams and their probs with the 2-letter history 'th'
f = open('th', 'w')
for key, value in probabilities_new.items():
  if key[0] == 't' and key[1] == 'h':
      f.write(str(key) + ' ' + str(value) + '\n')
f.close()

#generation
def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]

#array for generation
def generate(pattern):
    temp = dict()
    for key, value in probabilities.items():
        if key[:2]==pattern:
            temp[key] = value

    return weighted_values(temp.keys(), temp.values(), 1)[-1:]

#generate some output
f = open('generate', 'w')
output = tuple()
start = ('t','h','e')
output=start+output
while len(output)<300:
    start=output[-2:]
    l  = generate(start)
    output=output+tuple(l)

print ("Generated text: ")
print(''.join(output))

f.close()

#test data
file = 'test'

with open (file, "r") as myfile:
    test_data=myfile.read().lower().replace(" ", '')

test_data1 = re.sub("[^a-z]","",test_data)
eval = dict()
text_len = len(test_data1)

p = 1.0
#perplexity
for i in range(len(test_data1)-n+1):
    gram = tuple(data1[i:i+n])
    eval[gram] = probabilities_new[gram]
    p=math.pow(p*(1.0/eval[gram]),(1.0/text_len))

print('Perplexity: ')
print(p)
