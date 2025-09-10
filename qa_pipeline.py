from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

def run_pipeline():
    nlp_qa = pipeline("question-answering")

    result = nlp_qa(
        context="I’m Dhananjay, a Data Scientist with strong knowledge in Machine Learning, Deep Learning, and Computer Vision.",
        question="Who is Data Scientist?"
    )

    print("Pipeline Result:", result)

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-5-finetuned-squadv2")
    model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/bert-tiny-5-finetuned-squadv2")

    bert_tiny_nlp_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

    result2 = bert_tiny_nlp_qa(
        context="I’m Dhananjay, a Data Scientist with strong knowledge in Machine Learning, Deep Learning, and Computer Vision.",
        question="Who is Data Scientist?"
    )

    print("BERT Tiny Result:", result2)

if __name__ == "__main__":
    run_pipeline()
