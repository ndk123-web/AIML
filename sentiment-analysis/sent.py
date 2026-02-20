import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def sentiment_with_emoji(text):
    score = sia.polarity_scores(text)['compound']

    if score >= 0.6:
        return "😄 Very Positive"
    elif score >= 0.2:
        return "🙂 Positive"
    elif score > -0.2:
        return "😐 Neutral"
    elif score > -0.6:
        return "🙁 Negative"
    else:
        return "😡 Very Negative"

# Test
texts = [
    "I love this product so much!",
    "It's okay, not bad",
    "This is the worst experience ever",
    "I feel nothing about this"
]

for t in texts:
    print(t, "➡️", sentiment_with_emoji(t))