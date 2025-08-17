from safetensors.torch import load_file
from typing import List, Dict
import os
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer, models


class Matryoshka2dPooling(torch.nn.Module):
    """طبقة Matryoshka2dPooling المخصصة"""

    def __init__(
            self,
            word_embedding_dimension: int,
            dims: List[int],
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

        # تم التعديل هنا: استخدام models من sentence_transformers مباشرة
        self.base_pooling = models.Pooling(
            word_embedding_dimension,
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens
        )

        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(self.base_pooling.get_sentence_embedding_dimension(), d)
            for d in dims
        ])

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            self.activation = torch.tanh

        self._sentence_embedding_dimension = int(sum(dims))

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    def get_sentence_embedding_dimension(self) -> int:
        return self._sentence_embedding_dimension

    @staticmethod
    def load(input_path: str) -> 'Matryoshka2dPooling':
        """تحميل النموذج من المسار المحدد"""
        config_path = os.path.join(input_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model = Matryoshka2dPooling(
            word_embedding_dimension=config['word_embedding_dimension'],
            dims=config['dims'],
            normalize=config['normalize'],
            activation=config['activation']
        )

        weights_path = os.path.join(input_path, 'model.safetensors')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        return model


class FaissIndexer:
    def __init__(self, model_path: str, answers_path: str, faiss_dir: str, device: str = None):
        self.model_path = model_path
        self.answers_path = answers_path
        self.faiss_dir = faiss_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(self.model_path, device=self.device)

    def build_index(self):
        if not os.path.exists(self.answers_path):
            raise FileNotFoundError(f"ملف الإجابات غير موجود: {self.answers_path}")

        answers = []
        with open(self.answers_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    answers.append(data['answer'])
                except (json.JSONDecodeError, KeyError):
                    continue

        if not answers:
            raise ValueError("لا توجد إجابات صالحة")

        print(f"📦 حساب التضمينات لعدد {len(answers)} إجابة...")
        embeddings = self.model.encode(
            answers,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )

        # إنشاء مجلد FAISS
        os.makedirs(self.faiss_dir, exist_ok=True)

        # إنشاء الفهرس وحفظه
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # استخدام التشابه الكوني
        faiss.normalize_L2(embeddings)  # التطبيع
        index.add(embeddings)

        faiss.write_index(index, os.path.join(self.faiss_dir, "index.faiss"))

        # حفظ الإجابات
        with open(os.path.join(self.faiss_dir, "answers.jsonl"), 'w', encoding='utf-8') as f:
            for ans in answers:
                f.write(json.dumps({"answer": ans}, ensure_ascii=False) + "\n")

        print(f"✅ تم حفظ الفهرس والإجابات في {self.faiss_dir}")

if __name__ == "__main__":
    MODEL_DIR = "SemanticSearchModels/islamqa_retriever_acc_0.9275_ep_5_step_36"
    ANSWERS_FILE = "SemanticSearchData/answers/answers.jsonl"
    FAISS_DIR = "faiss_store"

    indexer = FaissIndexer(MODEL_DIR, ANSWERS_FILE, FAISS_DIR)
    indexer.build_index()