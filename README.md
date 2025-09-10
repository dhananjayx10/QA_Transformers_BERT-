# 📘 QA_Transformers_BERT

A simple **Question Answering system** using **Hugging Face Transformers** and **BERT** models. 🚀

This project demonstrates how to build a question-answering pipeline using Hugging Face’s `transformers` library. 
It uses **BERT (Bidirectional Encoder Representations from Transformers)** fine-tuned on **SQuAD v2** dataset for extracting answers from a given context.

---

## 🔹 Features
- Pre-trained **BERT-tiny** QA model from Hugging Face.
- Extracts answers from text using a simple `pipeline`.
- Lightweight and beginner-friendly implementation.
- Includes **performance benchmarking** with `%timeit`.

---

## 🔧 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/QA_Transformers_BERT.git
cd QA_Transformers_BERT
pip install -r requirements.txt
```

---

## 📜 Usage

### 1. Using Hugging Face Pipeline

```python
from transformers import pipeline

nlp_qa = pipeline("question-answering")

result = nlp_qa(
    context="I’m Dhananjay, a Data Scientist with strong knowledge in Machine Learning, Deep Learning, and Computer Vision.",
    question="Who is Data Scientist?"
)

print(result)
```

---

### 2. Using BERT-Tiny Fine-Tuned on SQuAD v2

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-5-finetuned-squadv2")
model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/bert-tiny-5-finetuned-squadv2")

bert_tiny_nlp_qa = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

result = bert_tiny_nlp_qa(
    context="I’m Dhananjay, a Data Scientist with strong knowledge in Machine Learning, Deep Learning, and Computer Vision.",
    question="Who is Data Scientist?"
)

print(result)
```

---

## ⚡ Benchmarking

You can also measure performance using Jupyter/IPython magic:

```python
%timeit nlp_qa(context="I’m Dhananjay, a Data Scientist...", question="Who is Data Scientist?")
```

---

## 📂 Project Structure

```
QA_Transformers_BERT/
│── README.md
│── requirements.txt
│── .gitignore
│── qa_pipeline.py
```

---

## 📊 Example Output

```json
{
  "score": 0.99,
  "start": 2,
  "end": 10,
  "answer": "Dhananjay"
}
```

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

---

## 📜 License
This project is licensed under the MIT License.
