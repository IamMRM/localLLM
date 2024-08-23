from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import os
from transformers import BitsAndBytesConfig
from llm_conversation import Conversation
from llama_cpp import Llama
import bitsandbytes as bnb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer=None


def load_model(model_name, model_dir="", use_safetensors=False):
    global model, tokenizer
    print(f"Loading model {model_name}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = LlamaForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="cpu",
        use_safetensors=use_safetensors,
        low_cpu_mem_usage=True,
        #offload_folder="offload_folder"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

def load_gguf_model(model_name, model_dir):
    global model
    print(f"Loading GGUF model {model_name}...")
    model = Llama(model_path=f"{model_dir}/{model_name}.gguf")
    print(f"Model {model_name} loaded successfully.")


def generate_text(prompt, conversation=None):
    max_length=500 # hardcoding because of less GPU VRAM
    if conversation:
        context = conversation.get_context()
        formatted_prompt = f"{context}\nHuman: {prompt}\nAI:"
    else:
        formatted_prompt = f"Human: {prompt}\nAI:"

    if tokenizer:
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=int(max_length)).to(model.device)
        attention_mask = inputs['attention_mask']
        input_ids = inputs['input_ids']
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=1024,  # Set a reasonable upper limit
                num_return_sequences=1, 
                do_sample=True,
                max_length=int(max_length),
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    else:
        response = model(formatted_prompt, max_tokens=500)
        decoded_output = response['choices'][0]['text']
    
    response = decoded_output.split("AI:")[-1].strip()

    if conversation:
        conversation.add_exchange(prompt, response)

    return response