import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def test_pytorch():
    print("Testing PyTorch:")
    print("Is CUDA available?", torch.cuda.is_available())
    x = torch.rand(5, 3)
    print("Tensor:", x)

def test_transformers():
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
    print(f"Sentiment: {sentiment}, Score: {score}")

def test_v():


    # Specify the desired model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Or choose another model

    # Create the pipeline with the specified model and device
    nlp = pipeline("sentiment-analysis", model=model_name, device=0)  # Use device=0 for GPU

    # Perform sentiment analysis
    result = nlp("we love you")
    print(result)


if __name__ == "__main__":
    #test_v()
    #test_pytorch()
    test_transformers()
