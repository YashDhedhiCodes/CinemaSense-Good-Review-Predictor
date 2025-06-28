
# 🎭 Sentiment Analysis using Simple RNN on IMDB Dataset

## 📌 Project Overview
This project uses a **Simple Recurrent Neural Network (SimpleRNN)** to perform sentiment analysis on movie reviews from the **IMDB dataset (50,000 reviews)** provided by `tensorflow.keras.datasets.imdb`.  
Unlike traditional binary classification, the model outputs a **sentiment score as a percentage**, reflecting the likelihood that a review is positive.

## 📂 Dataset
- **Source:** `tensorflow.keras.datasets.imdb`
- **Size:** 50,000 most frequent words used to encode reviews
- **Labels:**  
  - `1` → Positive review  
  - `0` → Negative review

## 🧠 Model Architecture
- **Embedding Layer** – Converts word indices to dense vector representations  
- **SimpleRNN Layer** – Captures sequential dependencies in review text  
- **Dense Output Layer** – Uses sigmoid activation to produce probability scores

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))
```

## 🏋️ Training Details
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 5–10 (tunable)  
- **Output:** Probability (e.g., 0.87 = 87% positive sentiment)

## 📊 Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Loss and Accuracy Curves  
- Threshold-based evaluation for classification

## 🚀 How to Run

### Prerequisites
- Python 3.8+  
- TensorFlow, Keras, NumPy, Matplotlib

### Steps
```bash
git clone https://github.com/yourusername/SimpleRNN-IMDB.git
cd SimpleRNN-IMDB
pip install -r requirements.txt
python main.py
```

## 📦 Files and Structure
```
SimpleRNN/
├── main.py               # Main training script
├── embedding.ipynb       # Text preprocessing & model building
├── prediction.ipynb      # Custom sentiment prediction using model
├── simple_rnn_imdb.h5    # Saved trained model
├── README.md
├── requirements.txt
```

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- IMDB dataset from keras.datasets  
- Matplotlib for visualization  

## 📌 Future Improvements
- Use LSTM or GRU for better sequence modeling  
- Add a web interface for interactive sentiment scoring  
- Explore pretrained embeddings like GloVe or Word2Vec
     
