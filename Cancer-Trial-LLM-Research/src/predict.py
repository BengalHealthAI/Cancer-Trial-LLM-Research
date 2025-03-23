import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "NousResearch/Llama-2-7b-chat-hf"
lora_weights = "./experiments/results"  # Path where fine-tuned weights are saved
device_map = {"": 0}

# Reload base model and merge LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map=device_map
)
model = PeftModel.from_pretrained(base_model, lora_weights)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def process_message(message):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)
    result = pipe(message)
    # Extract the assistant's output based on your formatting
    return result[0]['generated_text'].split('[/INST]')[1].strip()

if __name__ == '__main__':
    sample_message = "<s>[INST] <<SYS>>\nYour system message...\n<</SYS>>\n\nStudy Intervention: Example\nCondition: Example [/INST]"
    prediction = process_message(sample_message)
    print("Prediction:", prediction)
