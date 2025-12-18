import requests
from bs4 import BeautifulSoup
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from requests.exceptions import RequestException
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """Clean input text by removing non-ASCII characters and extra spaces."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)            # Replace multiple spaces with a single space
    return text.strip()



def fetch_front_page(url, tag='p', class_name=None, id_name=None):
    """Fetch and extract content from the front page of the provided URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        if class_name:
            elements = soup.find_all(tag, class_=class_name)
        elif id_name:
            elements = soup.find_all(tag, id=id_name)
        else:
            elements = soup.find_all(tag)

        text = ' '.join(element.get_text() for element in elements).strip()
        if not text:
            raise ValueError("No content found on the page")

        return clean_text(text)
    except (RequestException, ValueError) as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""

@lru_cache(maxsize=3)
def get_model_and_tokenizer(model_name):
    """Cache and return the tokenizer and model for a given model name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def analyze_sentiment_transformers(text, model_name):
    """Analyze sentiment using a pre-trained transformer model."""
    try:
        tokenizer, model = get_model_and_tokenizer(model_name)

        inputs = tokenizer(
            text[:2500],
            return_tensors='tf',
            truncation=True,
            padding=True,
            max_length=512
        )

        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(probs, axis=-1).numpy()[0]
        sentiment = model.config.id2label[predicted_class]
        score = probs[0][predicted_class].numpy()

        polarity = 1 if sentiment == 'POSITIVE' else -1
        return polarity * score
    except Exception as e:
        logging.error(f"Error analyzing sentiment with Transformers: {e}")
        return None

def fetch_all_urls(urls, tag='p', class_name=None, id_name=None):
    """Fetch content from all URLs in parallel."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda url: (url, fetch_front_page(url, tag, class_name, id_name)), urls))
    return dict(results)

def process_urls_for_texts(urls, texts, model_names, tag='p', class_name=None, id_name=None):
    """Process multiple URLs for given texts and compute sentiment scores using multiple models."""
    logging.info("Fetching content from URLs...")
    url_contents = fetch_all_urls(urls, tag, class_name, id_name)

    results = []
    for text in texts:
        text_scores = {}
        for model_name in model_names:
            model_scores = []
            for url, content in url_contents.items():
                if content:
                    combined_content = f"{text} {content}"
                    sentiment_score = analyze_sentiment_transformers(combined_content, model_name=model_name)
                    model_scores.append(sentiment_score)
                else:
                    model_scores.append(None)

            valid_scores = [score for score in model_scores if score is not None]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            text_scores[model_name] = {'average_sentiment_score': average_score}

        results.append({'text': text, 'model_scores': text_scores})

    return results

if __name__ == "__main__":
    urls1 = [
        "https://www.bloomberg.com",
        "https://www.cnbc.com",
        "https://www.ft.com",
        "https://finance.yahoo.com",
        "https://www.businessinsider.com",
        "https://www.forbes.com",
        "https://www.economist.com",
        "https://www.investopedia.com",
        "https://seekingalpha.com",
    ]

    urls2 = [
        "https://www.fool.com",
        "https://www.nyse.com",
        "https://www.tradingview.com",
        "https://www.investing.com",
        "https://uk.finance.yahoo.com",
        "https://ftalphaville.ft.com",
        "https://www.bloomberg.com/businessweek",
        "https://www.pimco.com",
        "https://www.capitaliq.com",
        "https://www.valueline.com",
        "https://www.hl.co.uk",
        "https://www.dividend.com",
        "https://www.moneycontrol.com",
        "https://www.home.saxo",
        "https://www.wealthmanagement.com",
        "https://www.wallstreetdaily.com",
        "https://www.jpmorgan.com",
        "https://www.goldmansachs.com",
        "https://www.morganstanley.com",
        "https://www.ubs.com"
    ]

    urls = urls1 + urls2

    texts = [
        "How is stock market doing today",
        "How is gold market doing today",
    ]

    model_names = [
        'cardiffnlp/twitter-roberta-base-sentiment',
        'nlptown/bert-base-multilingual-uncased-sentiment',
        'siebert/sentiment-roberta-large-english',
        'ProsusAI/finbert',
        'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
        'nickmuchi/distilroberta-finetuned-financial-text-classification',
        'SamLowe/roberta-base-go_emotions',
        'incredible45/News-Sentimental-model-Buy-Neutral-Sell',
        'finiteautomata/bertweet-base-sentiment-analysis',
        'ahmedrachid/FinancialBERT-Sentiment-Analysis',
        'tabularisai/multilingual-sentiment-analysis',
        'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
    ]

    results = process_urls_for_texts(urls, texts, model_names, tag='p', class_name=None, id_name=None)

    for result in results:
        print(f"Text: {result['text']}")
        print("Model Name - Sentiment Score Pairs:")
        for model_name, scores in result['model_scores'].items():
            print(f"  {model_name}: {scores['average_sentiment_score']}")
