from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import torch
import time

model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-12b",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

def prompt(s):
    input_ids = tokenizer(s, return_tensors="pt").input_ids.to('cuda')
    start = time.time()
    gen_tokens = model.generate(
      input_ids,
      do_sample=True,
      temperature=0.9,
      max_length=100
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    end = time.time()
    print("time to generate: ", end - start)
    return gen_text

# Change this prompt to be anything that you want.
print(prompt("Who was the first President of the United States?"))

