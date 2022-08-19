from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keras.preprocessing.text import tokenizer_from_json
from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences
from emoji import emojize
import pandas as pd
from tensorflow import keras
# classifier = keras.models.load_model('my_model.h5')
import numpy as np
import joblib
# from flair.data import Sentence

import pickle


# classifier = joblib.load('model.sav')
# classifier = pickle.load(open('model.pkl', 'rb'))
# tokenizer = Tokenizer()
def models():
    with open('models/tokenizer.json') as f:
        tokenizer = tokenizer_from_json(f.read())

    # Model reconstruction from JSON file
    with open('models/model_architecture.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('models/model_weights.h5')

    return model, tokenizer


classifier, tokenizer = models()

emoji_dict = {
    0: ":red_heart:",
    1: ":baseball:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}


def label_to_emoji(label):
    return emojize(emoji_dict[label])


app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('emoji_pred.html')


@app.route('/emojify', methods=['POST'])
def predict():
    text = request.form.get('text')
    if not len(text.strip()):
        return 'Dude!! enter something'
    print(text)
    data = [text]
    text_seq = tokenizer.texts_to_sequences(data)
    padded_text = pad_sequences(text_seq, maxlen=10, padding='post', truncating='post')
    emoji_pred = classifier.predict(padded_text)

    # emoji_pred1 = np.argmax(emoji_pred, axis=1)
    # print(emoji_pred1)
    # for i in range(len(data)):
    #     return label_to_emoji(emoji_pred1[i])
    emoji_pred1 = np.argsort(emoji_pred, axis=1)
    labels = list(emoji_pred1[0][-2:])
    print(list(emoji_pred1[0][-2:]))
    for i in labels:
        return label_to_emoji(i)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=True)
