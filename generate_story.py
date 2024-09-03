import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_story(prompt, model_path, max_length=100):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, do_sample=True)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return story

if __name__ == "__main__":
    prompt = "The hero entered the dark cave"
    model_path = "../models/storytelling_model.pth"
    story = generate_story(prompt, model_path)
    print(story)

