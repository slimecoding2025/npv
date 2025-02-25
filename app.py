# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل النموذج
model = tf.keras.models.load_model('models/turbo_ai_model.h5')

# تحميل tokenizer الخاص بالإجابات
with open('models/label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)

# تحميل tokenizer الخاص بالأسئلة (إعادة تدريب tokenizer بنفس البيانات المستخدمة في التدريب)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="<OOV>")
data = {
    'question': [
        "ما هو علاج الحمى؟",
        "ما هي أعراض الإنفلونزا؟",
        "ما هي الجرعة المناسبة للأسبرين؟"
    ]
}
tokenizer.fit_on_texts(data['question'])

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')

    # معالجة السؤال
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=20)

    # التنبؤ بالإجابة
    prediction = model.predict(padded_sequence)
    answer_index = np.argmax(prediction, axis=-1)[0]

    # تحويل الإجابة إلى نص
    if answer_index == 0:
        answer = "يمكن علاج الحمى باستخدام الباراسيتامول."
    elif answer_index == 1:
        answer = "تشمل أعراض الإنفلونزا الحمى والسعال وألم الجسم."
    elif answer_index == 2:
        answer = "الجرعة المناسبة تعتمد على السبب، لكنها غالبًا تتراوح بين 75-100 مجم يوميًا."
    else:
        answer = "عذرًا، لا أستطيع فهم سؤالك."

    return jsonify({'answer': answer})

# المسار الأساسي: الدردشة
@app.route('/healthz')
def chat():
    return render_template('chat.html')

# تشغيل الخادم
if __name__ == '__main__':
    app.run(debug=True)