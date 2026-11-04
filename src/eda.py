import json
from collections import Counter
# 4 se repeta in train cu id uri diferite 
# 15wmhj5 / 16256nv	a doua are in plus hopelessness
# mremnc / mtyysh
# 16hygfy / 16hygan	
# fiimgh / fiim31
# doua duplicate in train si test
# train
train_posts = []
with open("../data/train.json", encoding="utf-8") as jsonl:
    for line in jsonl:
        post = json.loads(line)
        train_posts.append(post)
train_text = []
train_label = []
for post in train_posts:
    train_text.append(post["text"])
    train_label.append(post["emotions"])
# test
test_posts = []
with open("../data/test.json", encoding="utf-8") as jsonl:
    for line in jsonl:
        post = json.loads(line)
        test_posts.append(post)
test_text = []
test_label = []
for post in test_posts:
    test_text.append(post["text"])
    test_label.append(post["emotions"])
# val
val_posts = []
with open("../data/val.json", encoding="utf-8") as jsonl:
    for line in jsonl:
        post = json.loads(line)
        val_posts.append(post)
val_text = []
val_label = []
for post in val_posts:
    val_text.append(post["text"])
    val_label.append(post["emotions"])
# print(len(set(train_text)& set(val_text)))
# print(len(set(train_text) & set(test_text)))
# print(len(set(test_text)& set(val_text)))
