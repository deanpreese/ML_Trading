import requests
from bs4 import BeautifulSoup
from transformers import pipeline, TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from requests.exceptions import RequestException

def fetch_front_page(url, tag='p', class_name=None, id_name=None):
    """Fetch and extract content from the front page of the provided URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract content based on the specified tag, class, and id
        if class_name:
            elements = soup.find_all(tag, class_=class_name)
        elif id_name:
            elements = soup.find_all(tag, id=id_name)
        else:
            elements = soup.find_all(tag)
        
        text = ' '.join(element.get_text() for element in elements).strip()
        
        if not text:
            raise ValueError("No content found on the page")
        
        return text
    except (RequestException, ValueError) as e:
        print(f"Error fetching {url}: {e}")
        return ""

def analyze_sentiment_transformers(text, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
    """Analyze sentiment using a pre-trained transformer model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(probs, axis=-1).numpy()[0]
        sentiment = model.config.id2label[predicted_class]
        score = probs[0][predicted_class].numpy()
        
        # Map label to polarity
        polarity = 1 if sentiment == 'POSITIVE' else -1
        return polarity * score
    except Exception as e:
        print(f"Error analyzing sentiment with Transformers: {e}")
        return None

def process_urls(urls, model_name, tag='p', class_name=None, id_name=None):
    """Process multiple URLs and compute sentiment scores."""
    results = []
    for url in urls:
        print(f"Processing {url}...")
        text = fetch_front_page(url, tag, class_name, id_name)
        if text:
            sentiment_score = analyze_sentiment_transformers(text, model_name=model_name)
            results.append({'url': url, 'sentiment_score': sentiment_score})
        else:
            results.append({'url': url, 'sentiment_score': None})
    return results

def analyze_sentiment_of_text(texts, model_name):
    """Analyze sentiment for a list of texts."""
    results = []
    for text in texts:
        sentiment_score = analyze_sentiment_transformers(text, model_name=model_name)
        results.append({'text': text, 'sentiment_score': sentiment_score})
    return results

if __name__ == "__main__":
    urls = [
        "https://www.bloomberg.com",
        "https://www.cnbc.com",
        "https://www.ft.com",
        "https://finance.yahoo.com",
        "https://www.businessinsider.com",
        "https://www.forbes.com",
        "https://www.economist.com",
        "https://www.investopedia.com",
        "https://seekingalpha.com",
        "https://www.fool.com",
        "https://www.nyse.com",
        "https://www.morningstar.com",
        "https://www.tradingview.com",
        "https://www.investing.com",
        "https://uk.finance.yahoo.com",
        "https://ftalphaville.ft.com",
        "https://www.bloomberg.com/businessweek",
        "https://www.spglobal.com",
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

    texts = [
        "The market is looking great today!",
        "There are some concerning trends in the economy.",
        "Investors are optimistic about the future.",
        "The company reported a significant loss this quarter.",
        "How is Gold doing today",
        "How is Bitcoin doing today",
        "What's the price of Gold"
    ]

    model_names = [
        #'distilbert-base-uncased-finetuned-sst-2-english',
        'roberta-base',
        #'bert-base-uncased'
    ]

    sentiment_results = []

    for model_name in model_names:
        print(f"Using model: {model_name}")

        # Analyze sentiment of texts directly
        print("Sentiment analysis of direct texts:")
        results_texts_transformers = analyze_sentiment_of_text(texts, model_name=model_name)
        
        sentiment_results.append(results_texts_transformers)
        
        for result in results_texts_transformers:
            print(result)

        # Use Transformers for sentiment analysis from URLs
        #print(f"Sentiment analysis from URLs using model: {model_name}")
        #results_transformers = process_urls(urls, model_name=model_name, tag='p', class_name=None, id_name=None)
        
        #sentiment_results.append(results_transformers)
        
        #for result in results_transformers:
        #   print(result)

        print("-------------------")

    df = pd.DataFrame(sentiment_results)
    print(df)
