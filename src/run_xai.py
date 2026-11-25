from pathlib import Path
from transformers import (AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer,)
import json
import torch 


# incarcam roberta fine tunat pt depressionemo
model_dir = Path("../models/deberta-depressionemo-best")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()
with open("../data/test.json", encoding="utf-8") as f:
    posts = [json.loads(line) for line in f if line.strip()]
with open("../models/results.json", encoding="utf-8") as f:
    thresholds = json.load(f)["thresholds"]

def score_post(text):
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        scores = torch.sigmoid(logits).tolist()
        return {
            model.config.id2label[i]: score
            for i, score in enumerate(scores)
        }

predictions =[]
for post_index, post in enumerate(posts):
    scores = score_post(post["text"])
    for label, score in scores.items():
        if score>= thresholds[label]:
            predictions.append({"post_index": post_index, "label": label, "original_score": score})

# fctie ii da lui qwen doar portiunea de text pe care a vazuto roberta, unde unele sunt putin mai lungi den256 de tokeni
def get_visible_text(text):
    encoding = tokenizer(text, truncation=True, max_length=256, return_offsets_mapping=True)
    end = max(stop for start, stop in encoding["offset_mapping"])
    return text[:end]

qwen_name = "Qwen/Qwen2.5-1.5B-Instruct"
qwen_device = "cuda"

qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_name,dtype=torch.float16,)
qwen_model.to(qwen_device)
qwen_model.eval()


with open("qwen_quotes.jsonl", "w", encoding="utf-8") as output_file:
    for prediction in predictions:
        label = prediction["label"]
        text = posts[prediction["post_index"]]["text"]
        visible_text = get_visible_text(text)
        prompt = (
            f"RoBERTa predicted the label: {label}\n"
            "Copy one short, continuous quote from the post that supports this label.\n"
            "Use the exact wording from the post, without paraphrasing.\n"
            'Return only JSON: {"quote": "exact text from the post"}.\n'
            'If there is no supporting quote, return {"quote": null}.\n\n'
            f"Post:\n{visible_text}"
        )
        messages = [{"role": "user", "content": prompt}]
        chat_text = qwen_tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        qwen_inputs = qwen_tokenizer(chat_text,return_tensors="pt",add_special_tokens=False).to(qwen_device)

        with torch.inference_mode():
            output_ids = qwen_model.generate(**qwen_inputs,max_new_tokens=96,do_sample=False)
        prompt_length = qwen_inputs["input_ids"].shape[1]
        answer_ids = output_ids[0, prompt_length:]
        answer = qwen_tokenizer.decode(answer_ids, skip_special_tokens=True)
        quote = None
        status = "invalid_response"

        try:
            response = json.loads(answer)
        except json.JSONDecodeError:
            response = None

        if isinstance(response, dict) and "quote" in response:
            candidate = response["quote"]
            if candidate is None:
                status = "no_quote"
            elif isinstance(candidate, str) and candidate.strip():
                if candidate in visible_text:
                    quote = candidate
                    status = "valid"
                else:
                    status = "quote_not_found"

        result = {
            **prediction,
            "quote": quote,
            "status": status,
            "raw_answer": answer,
        }
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_file.flush()