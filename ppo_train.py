import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import LoraConfig

#from trl.experimental.ppo import PPOConfig, PPOTrainer  # ✅ 推荐这样导入
from trl import PPOConfig, PPOTrainer  # ✅ 推荐这样导入

import torch
import trl.trainer.ppo_trainer as trl_ppo

# 先保存原始的 masked_whiten
_old_masked_whiten = trl_ppo.masked_whiten

def safe_masked_whiten(values, mask):
    # 如果这一小批里没有任何有效 token，就跳过归一化，直接返回原值
    if mask.sum() == 0:
        # 打个日志方便以后排查
        print("[WARN] masked_whiten: mask.sum() == 0, skip whitening for this minibatch.")
        return values
    return _old_masked_whiten(values, mask)

# 打补丁
trl_ppo.masked_whiten = safe_masked_whiten

policy_model_path = "./Qwen2.5-0.5B"
reward_model_path = "./Qwen2.5-0.5B-Reward"
dataset_path = "./COIG-P/data/*.parquet"

def main():
    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. policy/reference model
    # ✅ 去掉 device_map="auto" 让 accelerate 管设备
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        policy_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # 3. reward/value model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # 4. dataset
    def to_ppo_format(ex):
        messages = [{"role": "system", "content": "你是一个乐于助人的中文助手。"}]
        for t in ex["conversations"]:
            role = "user" if t["from"] == "human" else "assistant"
            messages.append({"role": role, "content": t["value"]})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        toks = tokenizer(prompt, truncation=True, max_length=512, padding=False)

        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }

    dataset = load_dataset("parquet", data_files=dataset_path)["train"]
    dataset = dataset.train_test_split(test_size=0.9, seed=32)["train"]
    dataset = dataset.train_test_split(test_size=0.2, seed=32)
    train_dataset = dataset["train"].map(to_ppo_format, num_proc=8, remove_columns=dataset["train"].column_names)
    test_dataset  = dataset["test"].map(to_ppo_format, num_proc=8, remove_columns=dataset["test"].column_names)
    def valid_example(ex):
        # 粗暴一点：至少要有 8 个 token 之类的
        return len(ex["input_ids"]) > 8

    train_dataset = train_dataset.filter(valid_example)
    test_dataset  = test_dataset.filter(valid_example)

    # 5. LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 6. PPOConfig
    training_args = PPOConfig(
        output_dir="./Qwen2.5-0.5B-PPO",
        learning_rate=1e-4,
        total_episodes=24000,

        batch_size=12,           # 全局 batch（所有 GPU 总和）
        mini_batch_size=4,       # 每次 PPO 更新用 4 个样本
        per_device_train_batch_size=12,
        gradient_accumulation_steps=2,

        num_ppo_epochs=1,
        logging_first_step=True,
        logging_steps=5,
        logging_strategy="steps",
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        gradient_checkpointing=True,
        report_to="tensorboard",
        logging_dir="./ppo_logs",
        bf16=torch.cuda.is_available(),
    )

    print("len(train_dataset) =", len(train_dataset))
    print("batch_size        =", training_args.batch_size)
    print("total_episodes    =", training_args.total_episodes)
    print("num_total_batches =", training_args.num_total_batches)

    # 7. PPOTrainer
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("./Qwen2.5-0.5B-PPO")
    tokenizer.save_pretrained("./Qwen2.5-0.5B-PPO")


if __name__ == "__main__":  # ✅ 多进程安全
    main()