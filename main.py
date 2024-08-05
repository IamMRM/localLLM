from transformers import AutoModel, AutoTokenizer
import torch

def load_model(model_name):
    # Load the model and tokenizer
    print(f"Loading model {model_name}...")
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
    model, tokenizer = load_model(model_name)
    
    # Example prompt
    prompt = "What is the future of artificial intelligence?"
    result = generate_text(model, tokenizer, prompt)
    
    print(f"Generated Text:\n{result}")

if __name__ == "__main__":
    main()
