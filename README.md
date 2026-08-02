# Visual Question Answering using BLIP

## Overview

This project demonstrates **Visual Question Answering (VQA)** using the **BLIP (Bootstrapping Language-Image Pre-training)** model from Hugging Face. The model takes an **image** and a **natural language question** as input and generates an answer based on the visual content of the image.

---

## Features

- Uses the pretrained **BLIP VQA Base** model.
- Processes both image and text inputs.
- Generates natural language answers.
- Performs inference without additional training.
- Built using **PyTorch** and **Hugging Face Transformers**.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Pillow (PIL)

---

## Model

**Model Name:**

```
Salesforce/blip-vqa-base
```

The model consists of:

- Vision Encoder
- Text Encoder
- Text Decoder

It combines image understanding with natural language processing to answer questions about images.

---

## Project Workflow

```
Image
   │
   ▼
Load Image (PIL)
   │
   ▼
Question
   │
   ▼
BLIP Processor
(Image + Question)
   │
   ▼
Pixel Values + Tokenized Question
   │
   ▼
BLIP VQA Model
   │
   ▼
Generate Answer Tokens
   │
   ▼
Decode Tokens
   │
   ▼
Final Answer
```

---

## Input

Example image:

```
data/
└── dog.png
```

Example question:

```text
What is the dog doing?
```

---

## Image and Text Processing

The processor automatically:

- Resizes the image
- Normalizes pixel values
- Converts the image to tensors
- Tokenizes the question
- Creates the required model inputs

```python
inputs = processor(image, question, return_tensors="pt")
```

---

## Model Inference

The model generates an answer using:

```python
with torch.no_grad():
    outputs = model.generate(**inputs)
```

`torch.no_grad()` disables gradient computation since the model is only being used for prediction.

---

## Decode the Output

The generated token IDs are converted back into readable text.

```python
answer = processor.tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```

---

## Example Output

```
Question : what is the dog doing?
Answer : playing
```

Depending on the image, possible answers include:

```
running
```

```
jumping
```

```
sleeping
```

```
sitting
```

---

## Project Structure

```
project/
│
├── data/
│   └── dog.png
│
├── blip_vqa.py
├── README.md
└── requirements.txt
```

---

## Installation

Install the required packages:

```bash
pip install torch torchvision transformers pillow
```

---

## How to Run

Run the script:

```bash
python blip_vqa.py
```

The program will:

1. Load the pretrained BLIP model.
2. Load the input image.
3. Process the image and question.
4. Generate an answer.
5. Display the question and predicted answer.

---

## Applications

- Visual Question Answering (VQA)
- AI-powered image assistants
- Accessibility tools for visually impaired users
- Intelligent image search
- Robotics and autonomous systems
- Human-computer interaction

---

## Future Enhancements

- Support multiple questions for the same image.
- Process multiple images in a batch.
- Add webcam support for real-time VQA.
- Build a web interface using Streamlit or Gradio.
- Deploy as a REST API using Flask or FastAPI.

---

## Author

**Shahid Farhan KP**
