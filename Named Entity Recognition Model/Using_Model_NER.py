import os
import re
import json
import torch
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import unicodedata

class NERProcessor:
    def __init__(self, model_path: str, symbols_to_remove: list[str], punctuation: list[str]):

        self.model_path = model_path
        self._load_model()
        self.symbols_to_remove = symbols_to_remove
        self.punctuation = punctuation

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ مجلد النموذج '{self.model_path}' غير موجود.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=512, truncation=True)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        except Exception as e:
            raise RuntimeError(f"❌ خطأ أثناء تحميل النموذج أو الـ Tokenizer: {str(e)}")

        self.ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="none",
            ignore_labels=["O"]
        )

    def clean_text(self, text):
        
        text = unicodedata.normalize("NFKC", text)
        # 1. إزالة التشكيل
        tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(tashkeel, '', text)

        # 2. توحيد الحروف العربية
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ؤ', 'و', text)
        text = re.sub(r'ئ', 'ي', text)
        text = re.sub(r'ة', 'ه', text)

        # 3. إزالة محتوى الأقواس والوسوم HTML
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        text = re.sub(r'\([^)]*\)', '', text)  # ( )
        text = re.sub(r'\[[^\]]*\]', '', text)  # [ ]
        text = re.sub(r'\{[^}]*\}', '', text)  # { }

        # 4. إزالة الرموز الغير عربية + مسافات
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

        # 5. إزالة رموز مخصصة
        for symbol in getattr(self, "symbols_to_remove", []):
            text = text.replace(symbol, '')

        # 6. معالجة علامات الترقيم
        for p in getattr(self, "punctuation", []):
            text = text.replace(p, f' {p} ')

        # 7. ضغط المسافات
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def split_sentences(self, text):
        return [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\؟|\!|\n)\s+', text) if s.strip()]

    def extract_entities(self, tokens, sentence):
        entities = []
        current = None

        for token in tokens:
            if token['word'].startswith('##'):
                if current:
                    current['word'] += token['word'][2:]
                    current['end'] = token['end']
                continue

            if token['entity'].startswith('B-'):
                if current:
                    entities.append(current)
                current = {
                    'word': token['word'],
                    'entity_group': token['entity'][2:],
                    'start': token['start'],
                    'end': token['end'],
                    'score': token['score']
                }
            elif token['entity'].startswith('I-') and current:
                if token['entity'][2:] == current['entity_group']:
                    if token['start'] > current['end']:
                        current['word'] += ' ' + token['word']
                    else:
                        current['word'] += token['word']
                    current['end'] = token['end']
                else:
                    entities.append(current)
                    current = None
            else:
                if current:
                    entities.append(current)
                current = None

        if current:
            entities.append(current)

        # تصفية الكيانات
        filtered = []
        for ent in entities:
            actual = sentence[ent['start']:ent['end']]
            clean = re.sub(r'^\W+|\W+$', '', actual).strip()
            if len(clean) >= 2 and ent['score'] >= 0.7 and not any(c.isdigit() for c in clean):
                ent['word'] = clean
                filtered.append(ent)

        return filtered

    def split_combined_entities(self, entities, sentence):
        result = []
        for ent in entities:
            text = ent['word']
            if ' و ' in text:
                parts = text.split(' و ')
                start = ent['start']
                for part in parts:
                    if part.strip():
                        new_ent = ent.copy()
                        new_ent['word'] = part.strip()
                        new_ent['start'] = start
                        new_ent['end'] = start + len(part)
                        result.append(new_ent)
                        start += len(part) + 3
            else:
                result.append(ent)
        return result

    def process_text(self, text):
        cleaned_text = self.clean_text(text)
        sentences = self.split_sentences(cleaned_text)
        all_entities = []

        for i, sentence in enumerate(sentences):
            try:
                tokens = self.ner_pipeline(sentence)
                entities = self.extract_entities(tokens, sentence)
                entities = self.split_combined_entities(entities, sentence)

                for ent in entities:
                    ent['sentence_index'] = i
                    all_entities.append(ent)
                    # print(f"✅ {ent['entity_group']}: '{ent['word']}' (ثقة: {ent['score']:.2f})")
            except Exception as e:
                print(f"⚠️ خطأ في الجملة رقم {i}: {str(e)}")
                continue

        # بناء النتيجة النهائية
        result = {
            "model_name": self.model_path,
            "all_entities": []
        }

        for i, sentence in enumerate(sentences):
            sentence_data = {
                "sentence": sentence,
                "entities": defaultdict(list)
            }
            for ent in all_entities:
                if ent['sentence_index'] == i:
                    sentence_data["entities"][ent['entity_group']].append(ent['word'])

            sentence_data["entities"] = dict(sentence_data["entities"])
            result["all_entities"].append(sentence_data)

        return result


def extract_entities_from_text(text):
    """
    تستقبل نص و NERProcessor، وتُرجع الكيانات بصيغة:
    [{'entity': '...', 'type': '...'}, ...]
    """
    symbols_to_remove = ['#', '*', '_', '=', '~', '\\', '/', "'", 'ـ', ';', '‘', '’', '"', '$', '%', '&', '+', '±', '÷',
                         '×', '√', '∞', '|', '^', '•', '§', '؛', '¶']
    punctuation = [',', '.', '،', '؟', '?', '!', ':', '«', '»', '[', ']', '{', '}']

    model_path = "Named Entity Recognition Data/ModelSaved/BERT_NER_MODEL"

    processor_ner = NERProcessor(model_path, symbols_to_remove=symbols_to_remove, punctuation=punctuation)

    ner_result = processor_ner.process_text(text)
    extracted_entities = []

    for entry in ner_result.get("all_entities", []):
        if entry["sentence"].strip() == text.strip():  # مطابقة الجملة
            for ent_type, ent_list in entry.get("entities", {}).items():
                for ent in ent_list:
                    extracted_entities.append({
                        "entity": ent,
                        "type": ent_type
                    })
    return extracted_entities


#
# def convert_entities_for_relation_extraction(text, entities):
#     """
#     تستقبل نص وكيانات (char-level) وتُرجع الكيانات بصيغة:
#     [{"start": token_start, "end": token_end, "type": "PER"}, ...]
#     لاستخدامها في مرحلة استخراج العلاقات.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
#     offsets = encoded["offset_mapping"]
#
#     final_entities = []
#     for ent in entities:
#         start_char = text.find(ent["entity"])
#         end_char = start_char + len(ent["entity"])
#
#         start_token = None
#         end_token = None
#
#         for idx, (start, end) in enumerate(offsets):
#             if start_token is None and start_char >= start and start_char < end:
#                 start_token = idx
#             if end_char > start and end_char <= end:
#                 end_token = idx + 1
#                 break
#
#         if start_token is not None and end_token is not None:
#             final_entities.append({
#                 "start": start_token,
#                 "end": end_token,
#                 "type": ent["type"]
#             })
#
#     return final_entities