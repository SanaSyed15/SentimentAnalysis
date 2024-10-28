from flask import Flask, request, jsonify, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define the main route for the web app
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route that interacts with the frontend
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the input data from the frontend
    text = data['text']  # Extract the text field
    sentiment_scores = sid.polarity_scores(text)  # Get sentiment scores
    
    # Determine sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
