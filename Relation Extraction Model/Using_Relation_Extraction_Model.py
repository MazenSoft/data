from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json

MODEL_DIR = "re_model_entity_marker3"


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

id2label = model.config.id2label
label2id = model.config.label2id

# دالة لإدخال علامات الكيانات في النص بنفس الطريقة أثناء التدريب
def insert_entity_markers(tokens, head_ent, tail_ent):
    tokens = tokens.copy()
    if head_ent["start"] < tail_ent["start"]:
        first_ent, second_ent = head_ent, tail_ent
        first_head_tag_open, first_head_tag_close = "[HEAD]", "[/HEAD]"
        second_tail_tag_open, second_tail_tag_close = "[TAIL]", "[/TAIL]"
    else:
        first_ent, second_ent = tail_ent, head_ent
        first_head_tag_open, first_head_tag_close = "[TAIL]", "[/TAIL]"
        second_tail_tag_open, second_tail_tag_close = "[HEAD]", "[/HEAD]"

    tokens.insert(second_ent["end"], second_tail_tag_close)
    tokens.insert(second_ent["start"], second_tail_tag_open)
    tokens.insert(first_ent["end"], first_head_tag_close)
    tokens.insert(first_ent["start"], first_head_tag_open)

    return " ".join(tokens)

# دالة لاستخدام النموذج على مثال جديد
def predict_relation(text_tokens, head_entity, tail_entity, model, tokenizer, id2label):
    """
    text_tokens: قائمة التوكنز للنص (مثلاً ["سارة", "تعيش", "في", "الدوحة", "."])
    head_entity, tail_entity: dict ب keys = start, end (مواقع التوكنز في النص)
    """
    # إضافة علامات الكيانات في النص
    input_text = insert_entity_markers(text_tokens, head_entity, tail_entity)

    # ترميز النص
    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # تنبؤ النموذج
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, pred_id].item()

    return id2label[pred_id], confidence

# مثال للاستخدام:
if __name__ == "__main__":
    # نص على شكل توكنز
    tokens = ["الفرس", ".", "بغداد", ".", "«", "الإنسان", "حيوان", "اجتماعي", "»", ".", "فتح", "مكة", "."]
    # الكيان الأول "فتح مكة" من توكن 10 إلى 12
    head_entity = {"start": 10, "end": 12}
    # الكيان الثاني "الفرس" من توكن 0 إلى 1
    tail_entity = {"start": 0, "end": 1}

    relation, confidence = predict_relation(tokens, head_entity, tail_entity, model, tokenizer, id2label)

    print(f"العلاقة المتوقعة: {relation} (ثقة: {confidence:.2f})")
