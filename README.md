This project builds a customer support automation system using classification models and Generative AI.
**Dataset**
- Variables: subject, body, answer, type, queue, priority, language, version, tag_1–tag_8
- Converted from dictionary format to a pandas DataFrame.
- Analysis focused on English text (dataset also contains German).
**Preprocessing**
- Tokenization: remove special characters, spaces, stop words, punctuation.
**Feature Engineering:**
- Keyword presence (refund, login issue, payment failed)
- Urgency words (urgent, asap, immediately)
- Sentiment scores (positive, negative, neutral and overall score)
**Scaling:** Sentiment scores scaled with MinMaxScaler for consistency.
- Text Features: TF‑IDF vectorization (train → val/test).
**Approaches**
1. Classical ML:
- TF‑IDF + engineered features → Logistic Regression, Gaussian NB, Linear SVM.
- Max F1 score: 51%.
- Sentence Transformer (all-MiniLM-L12-v2) + engineered features → same classifiers.
- Max F1 score: 32%.
3. Deep Learning:
- TensorFlow tokenizer + padding (max length = 100).
- Trainable Embedding layer → Bidirectional LSTM (dropout 0.3, recurrent dropout 0.3).
- Numeric features → Dense(32, ReLU).
- Concatenated → Dense(64, ReLU) + Dropout(0.3).
- Output → Dense(num_classes, Softmax).
- Loss: sparse_categorical_crossentropy, Optimizer: Adam, Metric: Accuracy.
- EarlyStopping (patience=5, restore best weights)
**Deployment**
- Model deployed in Streamlit.
- Numeric features scaled at inference.
- Integrated with Gemini AI to generate polite, automated customer responses.

