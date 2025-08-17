
import os
import random
import json
import logging
import re
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, InputExample, losses, evaluation, models
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset
from tqdm.autonotebook import trange
from typing import List, Tuple, Dict, Iterable, Callable
import os
import json
import torch
import torch.nn as nn
from sentence_transformers import models
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import json
import torch
import torch.nn as nn
from sentence_transformers import models



# إعدادات التسجيل المحسنة
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,  # تغيير من INFO إلى DEBUG
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('semantic_search_training.log')
    ]
)
logger = logging.getLogger(__name__)


class ArabicTextPreprocessor:
    """معالج النصوص العربية لتحسين جودة التضمين"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """توحيد النص العربي وإزالة الزوائد"""
        # إزالة التشكيل
        text = re.sub(r'[\u064B-\u065F]', '', text)

        # توحيد الهاء والتاء المربوطة
        text = re.sub(r'[ةه]', 'ه', text)

        # إزالة الأحرف المتكررة
        text = re.sub(r'(.)\1+', r'\1', text)

        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()

        # إزالة علامات الترقيم غير الضرورية
        text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)

        return text


class IslamQATripletDataset(Dataset):
    """مجموعة بيانات متخصصة لثلاثيات الأسئلة الدينية"""

    def __init__(self, file_path: str, max_samples: int = None, preprocess: bool = True):
        self.examples = []
        self.preprocess = preprocess
        self.load_data(file_path, max_samples)

    def load_data(self, file_path: str, max_samples: int):
        logger.info(f"جارٍ تحميل البيانات من {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    query = data["query"]
                    positive = data["positive"]
                    negative = data["negative"]

                    if self.preprocess:
                        query = ArabicTextPreprocessor.normalize_text(query)
                        positive = ArabicTextPreprocessor.normalize_text(positive)
                        negative = ArabicTextPreprocessor.normalize_text(negative)

                    self.examples.append(InputExample(
                        texts=[query, positive, negative],
                        label=1.0
                    ))
                except Exception as e:
                    logger.error(f"خطأ في معالجة السطر {i}: {str(e)}")
        logger.info(f"تم تحميل {len(self.examples)} ثلاثيات من {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Matryoshka2dPooling(torch.nn.Module):
    """
    بديل محلي لـ Matryoshka2dPooling
    """
    def __init__(
        self,
        word_embedding_dimension: int,
        dims: list,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        normalize: bool = True,
        activation: str = "tanh"
    ):
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.dims = dims
        self.normalize = normalize
        self.activation_name = activation

        self.base_pooling = models.Pooling(
            word_embedding_dimension,
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens
        )

        self.projections = nn.ModuleList([
            nn.Linear(self.base_pooling.get_sentence_embedding_dimension(), d)
            for d in dims
        ])

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            self.activation = torch.tanh

        self._sentence_embedding_dimension = int(sum(dims))

    def forward(self, features):
        pooled = self.base_pooling(features)
        sent_emb = pooled['sentence_embedding']
        outs = []
        for proj in self.projections:
            o = proj(sent_emb)
            o = self.activation(o)
            if self.normalize:
                denom = (o.norm(p=2, dim=1, keepdim=True) + 1e-9)
                o = o / denom
                outs.append(o)
        concat = torch.cat(outs, dim=1)
        return {'sentence_embedding': concat}

    def get_sentence_embedding_dimension(self):
        return self._sentence_embedding_dimension

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        config = {
            'word_embedding_dimension': self.word_embedding_dimension,
            'dims': self.dims,
            'normalize': self.normalize,
            'activation': self.activation_name
        }
        # استخدم نفس تنسيق حفظ sentence-transformers
        with open(os.path.join(output_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # استخدم safetensors بدلاً من torch.save
        from safetensors.torch import save_file
        save_file(self.state_dict(), os.path.join(output_path, 'model.safetensors'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'matryoshka2d_config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        model = Matryoshka2dPooling(
            word_embedding_dimension=config['word_embedding_dimension'],
            dims=config['dims'],
            normalize=config['normalize'],
            activation=config['activation']
        )
        state_dict = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state_dict)
        return model


class IslamQATrainer:
    def __init__(self, config: dict = None):
        # إعدادات التدريب
        # self.config = config or self.default_config()
        base_config = self.default_config()
        base_config.update(config or {})
        self.config = base_config

        self.setup_directories()
        self.setup_device()

        # تحميل النموذج
        self.model = self.load_model()
        self.best_score = -1.0

        # تسجيل المعلمات
        logger.info(f"معلمات التدريب: {json.dumps(self.config, indent=2, ensure_ascii=False)}")

    def default_config(self) -> dict:
        """إعدادات التدريب الافتراضية"""
        return {
            "model_name": "aubmindlab/bert-base-arabertv02",
            "output_dir": "SemanticSearchModels",
            "batch_size": 16,
            "epochs": 12,
            "warmup_ratio": 0.1,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_seq_length": 320,
            "evaluation_steps": 1000,
            "early_stopping_patience": 3,
            "use_matryoshka": True,
            "matryoshka_dims": [768, 512, 256, 128, 64],
            "use_amp": True,
            "max_train_samples": None,
            "max_eval_samples": 2000,
            "train_file": "SemanticSearchData/train_data.jsonl",
            "eval_file": "SemanticSearchData/test_data.jsonl",
            "corpus_file": "SemanticSearchData/answers.jsonl",
            "hard_negatives_ratio": 0.2,
            "gradient_accumulation_steps": 2
        }

    def setup_directories(self):
        """إنشاء المجلدات اللازمة"""
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(self.config["train_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(self.config["eval_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(self.config["corpus_file"]), exist_ok=True)

    def setup_device(self):
        """تحديد الجهاز المستخدم (GPU/CPU)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"جارٍ التشغيل على: {self.device}")

    def load_model(self) -> SentenceTransformer:
        """تحميل النموذج مع المعمارية المحسنة"""
        logger.info(f"جارٍ تحميل النموذج: {self.config['model_name']}")

        # طبقة تحويل النصوص
        word_emb = models.Transformer(
            self.config["model_name"],
            max_seq_length=self.config["max_seq_length"]
        )

        # طبقة التجميع
        pooling_model = models.Pooling(
            word_emb.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        # إضافة طبقة كثيفة لتحسين التمثيل
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=768,
            activation_function=torch.nn.Tanh()
        )

        # بعد إنشاء word_emb و pooling_model و dense_model كما لديك
        modules = [word_emb, pooling_model, dense_model]

        if self.config.get("use_matryoshka", False):
            matryoshka_model = Matryoshka2dPooling(
                word_embedding_dimension=word_emb.get_word_embedding_dimension(),
                dims=self.config.get("matryoshka_dims", [768, 512, 256]),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
                normalize=True,
                activation="tanh"
            )
            modules.append(matryoshka_model)

        return SentenceTransformer(modules=modules)

    def triplet_collate_fn(self, batch):
        anchors, positives, negatives = [], [], []

        for example in batch:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])

        # استخدم tokenizer النموذج بدلاً من self.tokenizer
        features = self.model.tokenize(anchors + positives + negatives)

        # تقسيم الميزات إلى ثلاث مجموعات
        batch_size = len(batch)
        anchor_features = {k: v[:batch_size] for k, v in features.items()}
        positive_features = {k: v[batch_size:2 * batch_size] for k, v in features.items()}
        negative_features = {k: v[2 * batch_size:] for k, v in features.items()}

        return [anchor_features, positive_features, negative_features], torch.ones(batch_size)

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = IslamQATripletDataset(
            self.config["train_file"],
            max_samples=self.config["max_train_samples"]
        )
        eval_dataset = IslamQATripletDataset(
            self.config["eval_file"],
            max_samples=self.config["max_eval_samples"]
        )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config["batch_size"],
            collate_fn=self.triplet_collate_fn
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=self.config["batch_size"],
            collate_fn=self.triplet_collate_fn
        )

        return train_dataloader, eval_dataloader

    def _extract_eval_scores(self, ir_scores):
        """Helper function to extract evaluation scores with version compatibility"""
        # Log all available keys for debugging
        logger.debug(f"Available evaluation keys: {list(ir_scores.keys())}")

        # Try different key formats for compatibility
        map_score = ir_scores.get('MAP@100', ir_scores.get('map', 0.0))
        recall_at_5 = ir_scores.get('Recall@5', ir_scores.get('recall@5', 0.0))

        return {
            'map': map_score,
            'recall@5': recall_at_5,
            'main_score': (map_score + recall_at_5) / 2
        }

    def _get_evaluation_metrics(self, ir_scores):
        """استخراج مقاييس التقييم مع التعامل مع تنسيقات المفاتيح المختلفة"""
        logger.debug(f"جميع مفاتيح التقييم المتاحة: {list(ir_scores.keys())}")

        # البحث عن MAP بأي تنسيق (مع/بدون بادئة islamqa-IR-eval)
        map_key = next((k for k in ir_scores.keys() if 'map' in k.lower()), None)
        map_score = ir_scores.get(map_key, 0.0) if map_key else 0.0

        # البحث عن Recall@5 بأي تنسيق
        recall_key = next((k for k in ir_scores.keys() if 'recall@5' in k.lower()), None)
        recall_at_5 = ir_scores.get(recall_key, 0.0) if recall_key else 0.0

        # تسجيل القيم المستخرجة للتأكد
        logger.debug(f"تم استخراج - MAP: {map_score}, Recall@5: {recall_at_5}")

        return {
            'map': float(map_score),
            'recall@5': float(recall_at_5),
            'main_score': (float(map_score) + float(recall_at_5)) / 2
        }
    def create_evaluator(self) -> evaluation.InformationRetrievalEvaluator:
        """إنشاء مقيم لاسترجاع المعلومات"""
        logger.info("جارٍ إنشاء مقيم استرجاع المعلومات...")

        # تحميل مجموعة التقييم
        queries = {}
        corpus = {}
        relevant_docs = {}

        with open(self.config["eval_file"], 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= self.config["max_eval_samples"]:
                    break
                data = json.loads(line)
                query_id = f"Q{idx}"
                queries[query_id] = ArabicTextPreprocessor.normalize_text(data["query"])
                corpus[f"D{idx}_pos"] = ArabicTextPreprocessor.normalize_text(data["positive"])
                corpus[f"D{idx}_neg"] = ArabicTextPreprocessor.normalize_text(data["negative"])
                relevant_docs[query_id] = [f"D{idx}_pos"]

        # إنشاء المقيم
        return evaluation.InformationRetrievalEvaluator(
            queries,
            corpus,
            relevant_docs,
            show_progress_bar=True,
            corpus_chunk_size=512,
            precision_recall_at_k=[3, 5, 10],
            name="islamqa-IR-eval"
        )

    def create_optimizer_scheduler(self, train_dataloader: DataLoader) -> Tuple[Optimizer, Callable]:
        """إنشاء محسن وجدولة معدل التعلم"""
        # حساب الخطوات
        num_train_steps = len(train_dataloader) * self.config["epochs"]
        warmup_steps = int(num_train_steps * self.config["warmup_ratio"])

        # إنشاء المحسن
        optimizer_params = {
            'lr': self.config["learning_rate"],
            'eps': 1e-6,
            'correct_bias': True,
            'weight_decay': self.config["weight_decay"]
        }
        optimizer = AdamW(self.model.parameters(), **optimizer_params)

        # جدولة معدل التعلم
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps
        )

        logger.info(f"عدد خطوات التدريب: {num_train_steps}")
        logger.info(f"خطوات الإحماء: {warmup_steps}")

        return optimizer, scheduler

    def save_model(self, score: float, epoch: int, step: int):
        """حفظ النموذج مع معلومات الأداء"""
        if score > self.best_score:
            self.best_score = score
            model_name = f"islamqa_retriever_acc_{score:.4f}_ep_{epoch}_step_{step}"
            save_path = os.path.join(self.config["output_dir"], model_name)

            self.model.save(save_path)

            # حفظ معلومات التدريب
            training_info = {
                "score": score,
                "epoch": epoch,
                "step": step,
                "date": datetime.now().isoformat(),
                "config": self.config
            }

            with open(os.path.join(save_path, "training_info.json"), "w", encoding="utf-8") as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 تم حفظ النموذج الأفضل في: {save_path} مع دقة: {score:.4f}")


    def train(self):
        """عملية التدريب الرئيسية"""
        logger.info("بدء التدريب...")

        # تحضير البيانات
        train_dataloader, eval_dataloader = self.create_dataloaders()
        ir_evaluator = self.create_evaluator()

        # إنشاء وظيفة الخسارة
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)

        # إعداد المحسن وجدولة معدل التعلم
        optimizer, scheduler = self.create_optimizer_scheduler(train_dataloader)

        # إعداد التدريب المتقدم
        self.model.to(self.device)

        # إعداد التوقف المبكر
        best_score = 0.0
        patience_counter = 0
        global_step = 0

        # حلقة التدريب
        for epoch in range(self.config["epochs"]):
            logger.info(f"════════════════════════════════════════════════════════════")
            logger.info(f"═══════════ العصر {epoch + 1}/{self.config['epochs']} ═══════════")
            logger.info(f"════════════════════════════════════════════════════════════")

            self.model.zero_grad()
            self.model.train()

            # شريط التقدم للعصر الحالي
            data_iterator = trange(
                len(train_dataloader),
                desc=f"العصر {epoch + 1}",
                mininterval=10
            )

            for step, batch in zip(data_iterator, train_dataloader):
                features, labels = batch
                labels = labels.to(self.device)

                # تحريك كل مجموعة مميزات إلى الجهاز
                features = [{
                    k: v.to(self.device) for k, v in f.items()
                } for f in features]

                # تمرير الأمامي
                loss_value = train_loss(features, labels)

                # التمرير الخلفي مع تجميع التدرجات
                loss_value.backward()

                # تحديث الأوزان مع تجميع التدرجات
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                # التقييم الدوري
                if global_step > 0 and global_step % self.config["evaluation_steps"] == 0:
                    logger.info(f"الخطوة {global_step}: التقييم...")

                    # تقييم IR
                    self.model.eval()
                    with torch.no_grad():
                        ir_scores = ir_evaluator(self.model)
                    self.model.train()

                    # استخراج النقاط
                    metrics = self._get_evaluation_metrics(ir_scores)

                    logger.info(f"نتائج التقييم (الخطوة {global_step}):")
                    logger.info(f"  - MAP: {metrics['map']:.4f}")
                    logger.info(f"  - Recall@5: {metrics['recall@5']:.4f}")
                    logger.info(f"  - النتيجة الرئيسية: {metrics['main_score']:.4f}")

                    # حفظ النموذج إذا تحسن الأداء
                    self.save_model(metrics['main_score'], epoch, global_step)

                    # التوقف المبكر
                    if metrics['main_score'] > best_score:
                        best_score = metrics['main_score']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        logger.info(
                            f"🛑 لم يتحسن الأداء. الصبر: {patience_counter}/{self.config['early_stopping_patience']}")
                        if patience_counter >= self.config["early_stopping_patience"]:
                            logger.info("⏹️ تم التوقف المبكر بسبب عدم التحسن")
                            return

            # التقييم في نهاية كل عصر
            logger.info(f"تقييم نهاية العصر {epoch + 1}...")
            self.model.eval()
            with torch.no_grad():
                ir_scores = ir_evaluator(self.model)
            self.model.train()

            # استخراج النقاط
            metrics = self._get_evaluation_metrics(ir_scores)

            logger.info(f"نتائج نهاية العصر {epoch + 1}:")
            logger.info(f"  - MAP: {metrics['map']:.4f}")
            logger.info(f"  - Recall@5: {metrics['recall@5']:.4f}")
            logger.info(f"  - النتيجة الرئيسية: {metrics['main_score']:.4f}")

            # حفظ النموذج
            self.save_model(metrics['main_score'], epoch, global_step)

            # التوقف المبكر
            if metrics['main_score'] > best_score:
                best_score = metrics['main_score']
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"🛑 لم يتحسن الأداء. الصبر: {patience_counter}/{self.config['early_stopping_patience']}")
                if patience_counter >= self.config["early_stopping_patience"]:
                    logger.info("⏹️ تم التوقف المبكر بسبب عدم التحسن")
                    return

        logger.info("✅ اكتمل التدريب!")

    @staticmethod
    def prepare_datasets():
        """إعداد مجموعات البيانات"""
        logger.info("جارٍ إعداد مجموعات البيانات...")

        # تقسيم البيانات
        IslamQATrainer.split_data(
            input_file="SemanticSearchData/generated_religious_examples.jsonl",
            train_file="SemanticSearchData/train_data.jsonl",
            test_file="SemanticSearchData/test_data.jsonl",
            train_ratio=0.9
        )

        # استخراج الإجابات
        IslamQATrainer.extract_and_save_positives(
            input_file="SemanticSearchData/generated_religious_examples.jsonl",
            output_file="SemanticSearchData/answers/answers.jsonl"
        )

    @staticmethod
    def split_data(input_file: str, train_file: str, test_file: str,
                   train_ratio: float = 0.9, seed: int = 42):
        """تقسيم البيانات إلى مجموعتي تدريب واختبار"""
        random.seed(seed)
        data = []

        logger.info(f"جارٍ تقسيم البيانات: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())

        random.shuffle(data)
        train_size = int(len(data) * train_ratio)

        # كتابة بيانات التدريب
        with open(train_file, 'w', encoding='utf-8') as f_train:
            for line in data[:train_size]:
                f_train.write(line + '\n')

        # كتابة بيانات الاختبار
        with open(test_file, 'w', encoding='utf-8') as f_test:
            for line in data[train_size:]:
                f_test.write(line + '\n')

        logger.info(f"تم إنشاء بيانات التدريب: {train_file} ({train_size} عينة)")
        logger.info(f"تم إنشاء بيانات الاختبار: {test_file} ({len(data) - train_size} عينة)")

    @staticmethod
    def extract_and_save_positives(input_file: str, output_file: str):
        """استخراج الإجابات الإيجابية وحفظها"""
        logger.info(f"جارٍ استخراج الإجابات من: {input_file}")

        answers = set()

        with open(input_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                item = json.loads(line)
                positive = ArabicTextPreprocessor.normalize_text(item.get("positive", "").strip())
                if positive and positive not in answers:
                    answers.add(positive)

        # حفظ الإجابات الفريدة
        with open(output_file, "w", encoding="utf-8") as f_out:
            for answer in answers:
                json.dump({"answer": answer}, f_out, ensure_ascii=False)
                f_out.write("\n")

        logger.info(f"✅ تم استخراج {len(answers)} إجابة فريدة في: {output_file}")



if __name__ == "__main__":
    # إعدادات مخصصة (اختيارية)
    custom_config = {
        "model_name": "aubmindlab/bert-base-arabertv02",
        "batch_size": 32,
        "epochs": 15,
        "learning_rate": 3e-5,
        "matryoshka_dims": [768, 512, 256, 128],
        "max_seq_length": 384
    }

    # إعداد البيانات
    IslamQATrainer.prepare_datasets()

    # بدء التدريب
    trainer = IslamQATrainer(custom_config)
    trainer.train()