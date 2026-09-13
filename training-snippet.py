import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# 1. Baseモデルとトークナイザーの準備
# Baseモデルは広範な「知識」を持つが、「対話のルール」や「指示に従う型」を知らない
model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. SFT用データの準備
# 知識を暗記させるためではなく、「Userの入力に対してAssistantとして振る舞う」関係性（Role）の学習用
raw_dataset = [
    {
        "messages": [
            {"role": "user", "content": "日本の首都はどこですか？"},
            {"role": "assistant", "content": "日本の首都は東京です。"},
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Pythonで現在時刻を取得するコードを教えて。",
            },
            {
                "role": "assistant",
                "content": "```python\nfrom datetime import datetime\nprint(datetime.now())\n```",
            },
        ]
    },
]


# 3. Chat Templateの適用
# 生テキストを「<|start_header_id|>user...」などの構造化トークン文字列へ変換し、モデルに役割の境界を提示する
def apply_template(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    }


dataset = Dataset.from_list(raw_dataset).map(apply_template)

# 4.【重要】「Assistantの応答部分のみ」を学習対象（Loss計算）にする設定
# Userの質問文（Prompt）は「丸暗記対象」から除外し（Loss計算上のラベルを -100 にマスク）、
# "assistant" ヘッダー以降の「応答の型と生成」のみを最適化対象として学習させる
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, tokenizer=tokenizer
)

# 5. LoRA（軽量微調整）の設定
# Baseモデルが持つ膨大な事前学習知識（重み）を破壊せず、対話制御のための追加重みのみを更新する
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 6. 学習設定
sft_config = SFTConfig(
    output_dir="./instruct_model_output",
    max_seq_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    bf16=True,
    dataset_text_field="text",
    packing=False,  # CompletionOnlyLM を適用するため False に設定
)

# 7. SFTTrainerの実行
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=collator,  # 応答生成部分のみを評価・学習するCollatorを注入
    processing_class=tokenizer,
)

trainer.train()
