import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score


def insert_entity_markers(tokens, head_ent, tail_ent):
    """
    إدخال علامات [HEAD], [/HEAD] و [TAIL], [/TAIL] حول الكيانات.
    head_ent و tail_ent لديهم 'start' و 'end' (مواقع التوكنز).
    """

    # نسخ القائمة لتعديلها
    tokens = tokens.copy()

    # لإضافة العلامات بدون إفساد الإزاحات، نبدأ من الكيان الثاني (الأبعد موقعياً) ثم الأول
    # ترتيب الإدراج عكسي حسب موقع البداية
    if head_ent["start"] < tail_ent["start"]:
        first_ent, second_ent = head_ent, tail_ent
        first_head_tag_open, first_head_tag_close = "[HEAD]", "[/HEAD]"
        second_tail_tag_open, second_tail_tag_close = "[TAIL]", "[/TAIL]"
    else:
        first_ent, second_ent = tail_ent, head_ent
        first_head_tag_open, first_head_tag_close = "[TAIL]", "[/TAIL]"
        second_tail_tag_open, second_tail_tag_close = "[HEAD]", "[/HEAD]"

    # إدخال العلامات للكيان الثاني (من النهاية أولاً)
    tokens.insert(second_ent["end"], second_tail_tag_close)
    tokens.insert(second_ent["start"], second_tail_tag_open)

    # إدخال العلامات للكيان الأول
    tokens.insert(first_ent["end"], first_head_tag_close)
    tokens.insert(first_ent["start"], first_head_tag_open)

    return " ".join(tokens)


def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # استخراج جميع أنواع العلاقات
    labels = sorted(set(rel["type"] for d in data for rel in d["relations"]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    dataset = []
    for d in data:
        tokens = d["tokens"]
        entities = d["entities"]
        relations = d["relations"]

        # لكل علاقة، ندرج علامات الكيانات داخل النص
        for rel in relations:
            head_ent = entities[rel["head"]]
            tail_ent = entities[rel["tail"]]

            # نص الجملة مع علامات تمييز الكيانات
            text_with_markers = insert_entity_markers(tokens, head_ent, tail_ent)

            dataset.append({
                "text": text_with_markers,
                "label": label2id[rel["type"]]
            })

    return Dataset.from_list(dataset), label2id, id2label


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1_macro,
        "f1_weighted": f1_weighted
    }



if __name__ == "__main__":
    DATA_PATH = "Relation Extraction model Data/relations_data.jsonl"
    MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
    OUTPUT_DIR = "re_model_entity_marker3"

    dataset, label2id, id2label = load_dataset(DATA_PATH)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # إضافة special tokens إلى tokenizer
    special_tokens = ["[HEAD]", "[/HEAD]", "[TAIL]", "[/TAIL]"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(special_tokens)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # تمديد حجم embedding ليتناسب مع الـ special tokens
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",              # تقييم كل حقبة
        save_strategy="epoch",                    # حفظ كل حقبة
        save_total_limit=1,                       # لا تحتفظ إلا بأفضل نموذج واحد فقط
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted", # استخدم f1 لتحديد الأفضل
        greater_is_better=True,                   # لأن F1 الأعلى أفضل
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.config.id2label = id2label
    model.config.label2id = label2id

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    preds = trainer.predict(tokenized["test"])
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)
    print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]))
