import os
import json
import shutil
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    TrainerCallback
)
import evaluate

class ModelTrainer:
    def __init__(self, file_path, config, progress_queue=None):
        self.MODEL_NAME = "CAMeL-Lab__bert-base-arabic-camelbert-mix-ner"
        self.OUTPUT_DIR = "Named Entity Recognition Data/ModelSaved/BERT_NER_MODEL"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.file_path = file_path
        self.config = config
        self.progress_queue = progress_queue

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = None
        self.tag_list = []
        self.label2id = {}
        self.id2label = {}

    def load_data(self) -> DatasetDict:
        dataset = load_dataset('json', data_files=self.file_path, split='train')
        return dataset.train_test_split(test_size=0.2)

    def prepare_model(self, dataset: DatasetDict):
        self.tag_list = sorted(set(tag for split in dataset for ex in dataset[split] for tag in ex["tags"]))
        self.label2id = {tag: i for i, tag in enumerate(self.tag_list)}
        self.id2label = {i: tag for i, tag in enumerate(self.tag_list)}

        self.model = self.load_or_create_model()
        self.model.to(self.DEVICE)

    def load_or_create_model(self):
        if os.path.exists(self.OUTPUT_DIR):
            try:
                return self.load_model_with_flexible_tags()
            except Exception as e:
                print(f"⚠️ خطأ في تحميل النموذج: {e}")

        return AutoModelForTokenClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(self.tag_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def load_model_with_flexible_tags(self):
        tag_file = os.path.join(self.OUTPUT_DIR, "tag_list.json")
        old_tag_list = []

        if os.path.exists(tag_file):
            with open(tag_file, "r", encoding="utf-8") as f:
                old_tag_list = json.load(f)

        if set(old_tag_list) == set(self.tag_list):
            return AutoModelForTokenClassification.from_pretrained(self.OUTPUT_DIR)

        new_label2id = {tag: i for i, tag in enumerate(self.tag_list)}
        new_id2label = {i: tag for i, tag in enumerate(self.tag_list)}

        model = AutoModelForTokenClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(self.tag_list),
            id2label=new_id2label,
            label2id=new_label2id
        )

        model_file = None
        for fname in os.listdir(self.OUTPUT_DIR):
            if fname.endswith(".safetensors") or fname.endswith(".bin"):
                model_file = os.path.join(self.OUTPUT_DIR, fname)
                break

        if model_file and os.path.exists(model_file):
            try:
                if model_file.endswith(".safetensors"):
                    from safetensors import safe_open
                    old_state_dict = {}
                    with safe_open(model_file, framework="pt") as f:
                        for key in f.keys():
                            old_state_dict[key] = f.get_tensor(key)
                else:
                    old_state_dict = torch.load(model_file)

                new_state_dict = model.state_dict()
                for name, param in old_state_dict.items():
                    if name in new_state_dict and param.size() == new_state_dict[name].size():
                        new_state_dict[name] = param

                model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"⚠️ خطأ أثناء نقل الأوزان: {e}")

        return model

    def tokenize_data(self, dataset: DatasetDict) -> DatasetDict:
        def tokenize_fn(examples):
            tokenized = self.tokenizer(
                examples["tokens"],
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True
            )

            all_labels = []
            for i, label_seq in enumerate(examples["tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.label2id[label_seq[word_idx]])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                all_labels.append(label_ids)

            tokenized["labels"] = all_labels
            return tokenized

        return dataset.map(tokenize_fn, batched=True)

    def compute_metrics(self, p):
        metric = evaluate.load("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_preds = [
            [self.tag_list[p] for (p, l) in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]

        true_labels = [
            [self.tag_list[l] for (p, l) in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]

        results = metric.compute(
            predictions=true_preds,
            references=true_labels,
            mode="strict",
            scheme="IOB2"
        )

        return {
            "eval_precision": results["overall_precision"],
            "eval_recall": results["overall_recall"],
            "eval_f1": results["overall_f1"],
            "eval_accuracy": results["overall_accuracy"],
        }

    def train_settings(self, tokenized_ds: DatasetDict):
        training_args = TrainingArguments(
            output_dir=self.OUTPUT_DIR,
            eval_strategy="epoch",
            learning_rate=self.config.get("learning_rate", 3e-5),
            per_device_train_batch_size=self.config.get("train_batch_size", 8),
            per_device_eval_batch_size=self.config.get("train_batch_size", 8),
            num_train_epochs=self.config.get("num_train_epochs", 10),
            weight_decay=self.config.get("weight_decay", 0.01),
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",  # المعيار اللي يقارن به
            greater_is_better=True
            # disable_tqdm = True
            # logging_dir="./logs",
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        progress_cb = ProgressCallback(progress_queue=self.progress_queue)

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[progress_cb]
        )

    def save_model(self):
        try:
            temp_dir = self.OUTPUT_DIR + "_temp"
            os.makedirs(temp_dir, exist_ok=True)
            self.model.save_pretrained(temp_dir)
            self.tokenizer.save_pretrained(temp_dir)

            with open(os.path.join(temp_dir, "tag_list.json"), "w", encoding="utf-8") as f:
                json.dump(self.tag_list, f, ensure_ascii=False, indent=2)

            if os.path.exists(self.OUTPUT_DIR):
                shutil.rmtree(self.OUTPUT_DIR)
            os.rename(temp_dir, self.OUTPUT_DIR)
        except Exception as e:
            print(f"❌ حدث خطأ أثناء حفظ النموذج: {e}")

    def run(self):
        dataset = self.load_data()
        self.prepare_model(dataset)
        tokenized_ds = self.tokenize_data(dataset)
        trainer = self.train_settings(tokenized_ds)
        trainer.train()
        results = trainer.evaluate()
        self.save_model()

        return results


class ProgressCallback(TrainerCallback):
    def __init__(self, progress_queue):
        self.progress_queue = progress_queue

    def on_step_end(self, args, state, control, **kwargs):
        progress = int((state.global_step / state.max_steps) * 100)
        if self.progress_queue:
            try:
                self.progress_queue.put(progress)
            except Exception as e:
                print(f"[ProgressCallback] ❌ خطأ أثناء إرسال التقدم: {e}")





if __name__ == "__main__":
    config = {
        "learning_rate": 3e-5,
        "train_batch_size": 8,
        "num_train_epochs": 5,
        "weight_decay": 0.01,
    }
    file_path = "Named Entity Recognition Data/long_ner_data.jsonl"  # مسار بيانات JSONL لديك

    trainer = ModelTrainer(file_path=file_path, config=config)
    results = trainer.run()
    print("✅ تم التدريب بنجاح! النتائج:")
    print(results)




# import json
#
# def split_data_to_jsonl(input_file, ner_file, re_file, semantic_file):
#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     with open(ner_file, "w", encoding="utf-8") as ner_f, \
#          open(re_file, "w", encoding="utf-8") as re_f, \
#          open(semantic_file, "w", encoding="utf-8") as sem_f:
#
#         for example in data:
#             # نكتب كل هيكل في ملفه بصيغة jsonl (كل سطر JSON مستقل)
#             ner_f.write(json.dumps(example["NER"], ensure_ascii=False) + "\n")
#             re_f.write(json.dumps(example["RE"], ensure_ascii=False) + "\n")
#             sem_f.write(json.dumps(example["SemanticSearch"], ensure_ascii=False) + "\n")
#
# if __name__ == "__main__":
#     input_path = "RE.json"       # ملف الإدخال بصيغة JSON (قائمة من الأمثلة)
#     ner_output = "NER.jsonl"
#     re_output = "RE.jsonl"
#     semantic_output = "SemanticSearch.jsonl"
#
#     split_data_to_jsonl(input_path, ner_output, re_output, semantic_output)
#     print("تم تقسيم البيانات إلى ملفات NER.jsonl و RED.jsonl و SemanticSearch.jsonl")
