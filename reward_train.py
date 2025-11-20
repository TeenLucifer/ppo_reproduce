import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

def make_conversation(example):
    prompt = example.data["conversations"][0]["value"]
    chosen = example.data["chosen"]["value"]
    rejected = example.data["rejected"]["value"]

    return {
        "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
        "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
    }

dataset = load_dataset("parquet", data_files="./COIG-P/data/*.parquet")
dataset = dataset.map(make_conversation)

print(dataset['train'][0])

## 1. 加载基模型（可以选用和 actor 相同架构的）
#model_name = "Qwen/Qwen2.5-0.5B"
#
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSequenceClassification.from_pretrained(
#    model_name,
#    num_labels=1,   # 奖励模型输出一个标量分数
#)
#
## 2. 配置训练参数
#config = RewardConfig(
#    output_dir="./rm_checkpoints",
#    per_device_train_batch_size=4,
#    learning_rate=1e-5,
#    num_train_epochs=2,
#)
#
## 3. 创建 Trainer
#trainer = RewardTrainer(
#    model=model,
#    tokenizer=tokenizer,
#    args=config,
#    train_dataset=your_preference_dataset,  # 这里放 HelpSteer3-Preference 处理后的数据
#)
#
#trainer.train()