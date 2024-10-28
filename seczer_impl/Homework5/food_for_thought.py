import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

token = "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"

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {token}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	print(response.headers)
	return response.json()

def format_zephyr_result():
	return """
        <|system|>
		{{preprompt}}</s>
		{{#each messages}}
            {{#ifUser}}<|user|>
            {{content}}</s>
            <|assistant|>
            {{/ifUser}}
			{{#ifAssistant}}{{content}}</s>
            {{/ifAssistant}}
		{{/each}}
    """

output = query({
	"inputs": "How many market crashes were there in the last 23 years?",
})

print(output)

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# checkpoint = "HuggingFaceH4/zephyr-7b-beta"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, load_in_4bit=True).to(device)  # You may want to use bfloat16 and/or move to GPU here

# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
#  ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
# print(tokenizer.decode(tokenized_chat[0]))

# outputs = model.generate(tokenized_chat, max_new_tokens=128) 
# print(tokenizer.decode(outputs[0]))