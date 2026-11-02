import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
prompt = "Artificial Intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
generated_text = " ".join(generated_text.split())
words = generated_text.split()
word_count = len(words)
character_count = len(generated_text)
sentences = re.split(r'[.!?]+', generated_text)
sentence_count = len([s for s in sentences if s.strip()])
keywords = ["artificial", "intelligence", "machine", "learning", "computer"]
keyword_count = {keyword: generated_text.lower().count(keyword) for keyword in keywords}
print("\nGENERATED TEXT\n----------------")
print(generated_text)
print("\nOUTPUT ANALYSIS\n----------------")
print("Number of words      :", word_count)
print("Number of characters:", character_count)
print("Number of sentences :", sentence_count)
print("\nKeyword Analysis\n----------------")
for keyword, count in keyword_count.items(): print(keyword, ":", count)
