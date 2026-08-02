from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

image = Image.open("data/dog.png").convert("RGB")
question = "what is the dog doing?"

inputs = processor(image, question, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs)

answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

print("question : ", question)
print("answer : ", answer)
