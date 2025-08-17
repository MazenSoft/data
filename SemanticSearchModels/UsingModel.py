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


class FaissRetriever:
    def __init__(self, model_path: str, faiss_dir: str, device: str = None):
        self.model_path = model_path
        self.faiss_dir = faiss_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(self.model_path, device=self.device)

        self.index = self._load_index()
        self.answers = self._load_answers()

    def _load_index(self):
        index_path = os.path.join(self.faiss_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"ملف الفهرس غير موجود: {index_path}")
        return faiss.read_index(index_path)

    def _load_answers(self):
        answers_path = os.path.join(self.faiss_dir, "answers.jsonl")
        if not os.path.exists(answers_path):
            raise FileNotFoundError(f"ملف الإجابات غير موجود: {answers_path}")

        answers = []
        with open(answers_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    answers.append(data['answer'])
                except:
                    continue
        return answers

    def search(self, query: str, top_k: int = 5, threshold: float = 0.70):
        query_emb = self.model.encode([query], convert_to_numpy=True, device=self.device)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            results.append({
                "answer": self.answers[idx],
                "score": float(score)
            })

        # ترتيب تنازلي
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

if __name__ == "__main__":
    MODEL_DIR = "SemanticSearchModels/islamqa_retriever_acc_0.9275_ep_5_step_36"
    FAISS_DIR = "faiss_store"

    retriever = FaissRetriever(MODEL_DIR, FAISS_DIR)

    questions = [
        "ما حكم من يتراخى عن صلاة الجمعة دون عذر؟",
        "كيفية التيمم للصلاة؟"
    ]

    for q in questions:
        print(f"\n❓ السؤال: {q}")
        results = retriever.search(q, top_k=5, threshold=0.70)
        if not results:
            print("⚠️ لا توجد نتائج أعلى من 70% تشابه")
        else:
            for r in results:
                print(f"🔹 ({r['score']:.2f}) {r['answer'][:80]}...")

