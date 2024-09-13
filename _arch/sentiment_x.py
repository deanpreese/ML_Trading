import requests
from bs4 import BeautifulSoup
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
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

def analyze_sentiment_transformers(text, model_name):
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

def process_urls_for_texts(urls, texts, model_names, tag='p', class_name=None, id_name=None):
    """Process multiple URLs for given texts and compute sentiment scores using multiple models."""
    url_contents = {}
    
    # Fetch content for each URL once
    for url in urls:
        print(f"Fetching content from {url}...")
        page_content = fetch_front_page(url, tag, class_name, id_name)
        url_contents[url] = page_content

    # Analyze sentiment for each text combined with each URL content using each model
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
            
            # Calculate average sentiment score for the text using the current model
            valid_scores = [score for score in model_scores if score is not None]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            text_scores[model_name] = {'average_sentiment_score': average_score, 'individual_scores': model_scores}
        
        results.append({'text': text, 'model_scores': text_scores})
    
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
        "How is Bitcoin doing today",
        "What's the price of Gold"
    ]

    model_names = [
        'distilbert-base-uncased-finetuned-sst-2-english',
        'roberta-base',
        'bert-base-uncased'
    ]

    # Analyze sentiment of texts based on content from URLs using multiple models
    results = process_urls_for_texts(urls, texts, model_names, tag='p', class_name=None, id_name=None)

    for result in results:
        print(result)
