import json
import re
import requests
import time
import os
from typing import Optional, Union, List

class GPTProcessor:
    API_URL = "https://api.groq.com/openai/v1/chat/completions"  # عدل حسب مزودك
    DEFAULT_MODEL = "llama3-70b-8192"

    @staticmethod
    def call_gpt(
        system_prompt: str,
        user_prompt: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
        max_tokens: int = 200,  # قللتها للإجابة المختصرة كما طلبت
        retries: int = 3
    ) -> str:

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        for attempt in range(retries):
            try:
                response = requests.post(GPTProcessor.API_URL, headers=headers, json=data)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    print("⏳ تم الوصول للحد الأقصى، إعادة المحاولة بعد تأخير 10 ثواني...")
                    time.sleep(10)
                else:
                    print(f"خطأ في الاستجابة {response.status_code}: {response.text}")
                    time.sleep(5)
            except Exception as e:
                print(f"استثناء أثناء الاتصال: {str(e)}")
                time.sleep(5)
        raise Exception("فشل الاتصال بالنموذج بعد عدة محاولات.")

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
        text = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]', '', text)
        return text.strip()

    @staticmethod
    def generate_example(api_key: str, model: str = DEFAULT_MODEL) -> dict:
        system_prompt = (
            "أنت خبير في النصوص الدينية الشرعية، وتجيب فقط باللغة العربية الفصحى. "
            "لا تستخدم أي كلمة أو جملة باللغة الإنجليزية أبداً في السؤال أو في الإجابات. "
            "من فضلك، أنشئ مثالاً واحداً بصيغة JSON فقط (دون أي شرح إضافي) يحتوي على ثلاثة حقول:\n"
            "1) query: سؤال ديني واضح ومحدد باللغة العربية فقط.\n"
            "2) positive: إجابة صحيحة شرعية مختصرة أو متوسطة الطول باللغة العربية فقط.\n"
            "3) negative: إجابة خاطئة أو مغلوطة مختصرة أو متوسطة الطول باللغة العربية فقط.\n"
            "احرص على التنويع في الأسئلة والأجوبة ولا تكرر الأمثلة. "
            "الإجابة الخاطئة تكون تدور على نفس الموضوع."
        )

        user_prompt = "يرجى توليد مثال واحد."

        raw_output = GPTProcessor.call_gpt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            model=model,
            temperature=0.85,
            max_tokens=600
        )

        cleaned_output = GPTProcessor.clean_text(raw_output)

        try:
            example = json.loads(cleaned_output)
        except Exception:
            # إذا لم يستطع التحويل إلى JSON، نعيد مثال فارغ مع النص الخام في الإجابة الصحيحة:
            example = {
                "query": "خطأ في توليد السؤال",
                "positive": cleaned_output,
                "negative": ""
            }
        return example

def main():
    api_keys = [

        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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"
    ]

    output_file = "generated_religious_examples.jsonl"
    num_examples = 10000

    generated_questions = set()
    current_api_key_index = 0

    # قراءة الملف الحالي إذا وجد لجمع الأسئلة لتجنب التكرار
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    question = item.get("query", "").strip()
                    if question:
                        generated_questions.add(question)
                except:
                    continue

    count = len(generated_questions)
    print(f"تم العثور على {count} أمثلة موجودة مسبقاً، سيتم توليد {num_examples - count} أمثلة جديدة.")

    with open(output_file, "a", encoding="utf-8") as f:
        while count < num_examples:
            api_key = api_keys[current_api_key_index]
            try:
                example = GPTProcessor.generate_example(api_key)
                question = example.get("query", "").strip()

                # تفادي التكرار أو الأسئلة الفارغة أو الأخطاء
                if question in generated_questions or question == "" or question.startswith("خطأ"):
                    print("⚠️ مثال مكرر أو غير صالح، يعاد المحاولة...")
                    continue

                generated_questions.add(question)
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                f.flush()

                count += 1
                print(f"✅ تم توليد مثال رقم {count}: {question}")

            except Exception as e:
                print(f"❌ حدث خطأ: {e}")
                # التبديل بين مفاتيح API عند الخطأ (اختياري)
                current_api_key_index = (current_api_key_index + 1) % len(api_keys)
                print(f"🔁 التبديل إلى مفتاح API رقم {current_api_key_index+1}")
                time.sleep(10)  # انتظر قبل إعادة المحاولة

if __name__ == "__main__":
    main()
