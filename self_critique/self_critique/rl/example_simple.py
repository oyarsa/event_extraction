import torch
from transformers import AutoTokenizer
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)
from trl.core import respond_to_batch

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<PAD>"})

# initialize trainer
ppo_config = PPOConfig(batch_size=1)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

print("Query tensor")
print(query_tensor)
output = model(query_tensor)
print("Model output")
print(output)
# get model response
response_tensor = respond_to_batch(model, query_tensor)
print("Response tensor")
print(response_tensor)
print("end response tensor")

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
ppo_trainer.log_stats(
    train_stats, {"query": [query_tensor], "response": [response_tensor]}, reward
)
