import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)
    except Exception as e:
        print(f"Error analyzing sentiment with TextBlob: {e}")
        return None

def analyze_sentiment_transformers(text, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
    """Analyze sentiment using a pre-trained transformer model with PyTorch."""
    try:
        print("Testing Transformers:")
        model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        text = "The market is looking great today!"
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        sentiment = model.config.id2label[predicted_class]
        score = probs[0][predicted_class].item()
        
        # Custom scoring logic
        #if sentiment == 'POSITIVE':
        #    polarity = 1
        #elif sentiment == 'NEGATIVE':
        #    polarity = -1
        #else:
        #    polarity = 0  # Assuming 'NEUTRAL' or other sentiment labels

        #return polarity * score
        return score
    
    except Exception as e:
        print(f"Error analyzing sentiment with Transformers: {e}")
        return None

def process_urls(urls, use_transformers=False, tag='p', class_name=None, id_name=None):
    """Process multiple URLs and compute sentiment scores."""
    results = []
    for url in urls:
        print(f"Processing {url}...")
        text = fetch_front_page(url, tag, class_name, id_name)
        if text:
            if use_transformers:
                sentiment_score = analyze_sentiment_transformers(text)
            else:
                sentiment_score = analyze_sentiment_textblob(text)
            
            results.append({'url': url, 'sentiment_score': sentiment_score})
        else:
            results.append({'url': url, 'sentiment_score': None})
    return results

def analyze_sentiment_of_text(texts, use_transformers=False):
    """Analyze sentiment for a list of texts."""
    results = []
    for text in texts:
        if use_transformers:
            sentiment_score = analyze_sentiment_transformers(text)
        else:
            sentiment_score = analyze_sentiment_textblob(text)
        
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
        "https://www.thestreet.com",
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
        "The company reported a significant loss this quarter."
    ]

    # Analyze sentiment of texts directly
    print("Sentiment analysis of direct texts:")
    results_texts_textblob = analyze_sentiment_of_text(texts, use_transformers=False)
    for result in results_texts_textblob:
        print(result)

    print(" ")
    results_texts_transformers = analyze_sentiment_of_text(texts, use_transformers=True)
    for result in results_texts_transformers:
        print(result)


    print(" ")
    # Use TextBlob for sentiment analysis from URLs
    print("Results using TextBlob:")
    results_textblob = process_urls(urls, use_transformers=False, tag='p', class_name=None, id_name=None)
    for result in results_textblob:
        print(result)
    
    # Use Transformers for sentiment analysis from URLs
    print("Results using Transformers:")
    results_transformers = process_urls(urls, use_transformers=True, tag='p', class_name=None, id_name=None)
    for result in results_transformers:
        print(result)
