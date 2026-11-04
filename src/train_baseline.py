import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import plotly.express as px
from sklearn.metrics import precision_recall_fscore_support
import joblib

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

mlb = MultiLabelBinarizer()
mlb.fit(train_label)
y_train, y_val, y_test = mlb.transform(train_label), mlb.transform(val_label), mlb.transform(test_label) 
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(train_text)
x_val, x_test = vectorizer.transform(val_text), vectorizer.transform(test_text)
print(x_train.shape)
clf = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=1000, class_weight="balanced")).fit(x_train, y_train)
y_pred = clf.predict(x_val)
y_prob = clf.predict_proba(x_val)
#####  0.45 ales pe val prin maximizarea macro f1  ###################################################
y_pred_45 = (y_prob >= 0.45).astype(int)
# print(y_pred.shape)
# print(y_pred[0])
# print("micro f1 = ", f1_score(y_val, y_pred_45, average="micro"))
# print("macro f1 = ", f1_score(y_val, y_pred_45, average="macro"))
# print(classification_report(y_val, y_pred_45, target_names=mlb.classes_, zero_division=0))
y_prob_test = clf.predict_proba(x_test)
y_pred_test = (y_prob_test >=0.45).astype(int)
print("micro f1 = ", f1_score(y_test, y_pred_test, average="micro"))
print("macro f1 = ", f1_score(y_test, y_pred_test, average="macro"))
print(classification_report(y_test, y_pred_test, target_names=mlb.classes_, zero_division=0))

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, zero_division=0)
fig = px.bar(x=mlb.classes_,y=f1,title="F1 per emotion", labels={"x":"Emotion", "y":"F1 score"}, text_auto=".2f")
fig.update_yaxes(range=[0,1])
fig.write_image("../figures/f1_emotion_baseline.png", scale=2)
joblib.dump(clf, "../models/classifier.joblib")
joblib.dump(vectorizer, "../models/tfidf.joblib")
joblib.dump(mlb, "../models/labels.joblib")