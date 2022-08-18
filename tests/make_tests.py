from transformers import T5Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import requests

model_id = "t5-base"
num_tests = 1_000

tokenizer = T5Tokenizer.from_pretrained(model_id, model_max_length=512)

data = load_dataset("opus_books", "en-fr")

tests = []
test_srcs = set()
for test_index in tqdm(range(num_tests)):
    translation = data["train"][test_index]["translation"]
    for lang in ["en", "fr"]:
        src = translation[lang]
        if src not in test_srcs:
            test_srcs.add(src)
            encoded = tokenizer.encode(src)
            decoded = tokenizer.decode(encoded)
            tests.append((src, encoded, decoded))

jtests = json.dumps(tests)
with open(f"{model_id}-tests.json", "w") as f:
    f.write(jtests)

jtokenizer_url = "https://huggingface.co/t5-base/raw/main/tokenizer.json"
r = requests.get(jtokenizer_url)
jtokenizer = r.text
with open(f"{model_id}-tokenizer.json", "w") as f:
    f.write(jtokenizer)
