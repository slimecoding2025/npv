# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import os

# عينة من البيانات
data = {
    'question': [
        "ما هو علاج الحمى؟",
        "ما هي أعراض الإنفلونزا؟",
        "ما هي الجرعة المناسبة للأسبرين؟"
    ],
    'answer': [
        "يمكن علاج الحمى باستخدام الباراسيتامول.",
        "تشمل أعراض الإنفلونزا الحمى والسعال وألم الجسم.",
        "الجرعة المناسبة تعتمد على السبب، لكنها غالبًا تتراوح بين 75-100 مجم يوميًا."
    ]
}

df = pd.DataFrame(data)

# معالجة النصوص
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['question'])
sequences = tokenizer.texts_to_sequences(df['question'])
padded_sequences = pad_sequences(sequences, padding='post', maxlen=20)

# تحويل الإجابات إلى تسلسلات رقمية
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(df['answer'])
label_sequences = label_tokenizer.texts_to_sequences(df['answer'])
padded_labels = pad_sequences(label_sequences, padding='post', maxlen=50)

# إنشاء النموذج
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(len(label_tokenizer.word_index) + 1, activation='softmax')
])

# تدريب النموذج
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, padded_labels, epochs=100)

# حفظ النموذج في مجلد models
os.makedirs('models', exist_ok=True)  # إنشاء مجلد models إذا لم يكن موجودًا
model.save('models/turbo_ai_model.h5')

# حفظ tokenizer الخاص بالإجابات
import pickle
with open('models/label_tokenizer.pkl', 'wb') as f:
    pickle.dump(label_tokenizer, f)