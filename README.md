# Movie_review_model 
# Movie Review Sentiment Analysis using LSTM

## 🎬 Project Overview
This project implements a deep learning model for sentiment analysis on movie reviews using Long Short-Term Memory (LSTM) networks. The model can classify movie reviews as either positive or negative, making it useful for automated sentiment analysis of film critiques and audience reactions.

## 🎯 Features
- Text preprocessing pipeline for movie reviews
- LSTM-based deep learning model
- Word embedding layer for text vectorization
- Binary sentiment classification (positive/negative)
- Model evaluation metrics and visualization

## 🛠️ Technical Architecture
The model consists of three main layers:
1. **Embedding Layer**: Converts word indices into dense vectors of fixed size (128 dimensions)
2. **LSTM Layer**: Processes sequential data with 128 units and dropout for regularization
3. **Dense Layer**: Final classification layer with sigmoid activation for binary output

```python
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- NLTK

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/movie-review-sentiment.git

# Navigate to project directory
cd movie-review-sentiment

# Install required packages
pip install -r requirements.txt
```

## 📊 Dataset
The model can be trained on various movie review datasets such as:
- IMDb Movie Reviews Dataset
- Rotten Tomatoes Reviews
- Amazon Movie Reviews

Ensure your data is preprocessed and formatted as follows:
- Text reviews cleaned and tokenized
- Labels converted to binary format (0 for negative, 1 for positive)
- Reviews padded to uniform length (200 words in current configuration)

## 💻 Usage
```python
# Example code for prediction
from model.predictor import SentimentPredictor

predictor = SentimentPredictor()
review = "This movie was absolutely fantastic! Great performance by all actors."
sentiment = predictor.predict(review)
print(f"Sentiment: {'Positive' if sentiment > 0.5 else 'Negative'}")
```

## 📈 Performance
- Training Accuracy: ~93%
- Validation Accuracy: ~91%
- F1 Score: ~0.90

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


```
