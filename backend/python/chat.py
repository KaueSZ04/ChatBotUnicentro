import nltk
from nltk.stem import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')

stemmer = LancasterStemmer()

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
training = []
output = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
random.shuffle(training)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])
tf.keras.backend.clear_session()

model = Sequential()
model.add(Dense(8, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=5000, batch_size=8, verbose=1)

model.save('model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return np.array(bag)

def response(user_input):
    p = bow(user_input, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return_list = sorted(return_list, key=lambda x: x['probability'], reverse=True)
    response_intent = return_list[0]['intent']

    for intent in intents['intents']:
        if intent['tag'] == response_intent:
            responses = intent['responses']

    return random.choice(responses)

training_data = {'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}
with open("training_data.pickle", "wb") as f:
    pickle.dump(training_data, f)

while True:
    pergunta = input("Voce - ")
    if pergunta=="parar": break
    else:
        print(response(pergunta))