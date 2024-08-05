from transformers import LlamaForCausalLM, AutoTokenizer
import torch

def load_model(model_name, device):
    print(f"Loading model {model_name}...")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == torch.device("cuda") else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    attention_mask = inputs['attention_mask']
    input_ids = inputs['input_ids']
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, pad_token_id=pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_name, device)
    
    prompt = "Q: What do you think is the most beautiful thing about being alive?\nA:"
    result = generate_text(model, tokenizer, prompt, max_length=1000)
    
    print(f"Generated Text:\n{result}")

if __name__ == "__main__":
    main()
