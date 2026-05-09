# app.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model setup (English to Hindi)
model_name = "Helsinki-NLP/opus-mt-en-hi"

# tokenizer for model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# seqmodel for model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tera input text
text = "I am a software engineer building AI systems."

# Magic happens here
inputs = tokenizer(text, return_tensors="pt")
translated_tokens = model.generate(**inputs)
result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print("\n--- Translation Ready ---")
print("English:", text)
print("Hindi:", result)
