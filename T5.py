import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
import numpy as np
import re
import os

# --- 配置 ---
MODEL_NAME = "Langboat/mengzi-t5-base" # 中文T5模型 (~250M params)
# MODEL_NAME = "google/mt5-base" # 多语言模型 (~580M params)

TRAIN_FILE = "train.json"
TEST_FILE = "test1.json"
OUTPUT_DIR = "./models"
PREDICTIONS_FILE = "predictions444.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练超参数设置
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SAVE_STEPS = 500




def load_data(train_path, test_path):

    train_data = []
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            # 确保字段完整
            train_data = [{'id': item['id'], 'content': item['content'], 'output': item['output']}
                          for item in train_data if 'content' in item and 'output' in item]
    else:
        print(f"Warning: Training file {train_path} not found.")

    test_data = []
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

            test_data = [{'id': item['id'], 'content': item['content']}
                         for item in test_data if 'content' in item]
    else:
        print(f"Warning: Test file {test_path} not found.")

    return train_data, test_data

# 数据预处理
def preprocess_data(examples, tokenizer):
    # 对输入和目标文本进行分词处理
    inputs = [ex for ex in examples['content']]
    targets = [ex for ex in examples['output']]


    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)


    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)

    #  将padding token转换为 -100，使其在计算loss时被忽略
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    print("--- Starting Hate Speech Quadruple Extraction Task ---")
    print(f"输出目录是否存在: {os.path.exists(OUTPUT_DIR)}")
    # 1. 加载训练和测试数据
    print(f"\n1. Loading data from {TRAIN_FILE} and {TEST_FILE}...")
    train_examples, test_examples = load_data(TRAIN_FILE, TEST_FILE)

    if not train_examples:
        print("Error: No training data loaded. Exiting.")
        exit()
    if not test_examples:
        print("Warning: No test data loaded. Will only train.")


    # 构造 Hugging Face 的 Dataset
    train_dict = {'id': [e['id'] for e in train_examples],
                  'content': [e['content'] for e in train_examples],
                  'output': [e['output'] for e in train_examples]}
    train_dataset = Dataset.from_dict(train_dict)

    if test_examples:
        test_dict = {'id': [e['id'] for e in test_examples],
                     'content': [e['content'] for e in test_examples]}

        test_dict['output'] = [""] * len(test_examples)
        test_dataset = Dataset.from_dict(test_dict)
        raw_datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})
    else:
         raw_datasets = DatasetDict({'train': train_dataset})

    print(f"   Loaded {len(train_dataset)} training examples.")
    if 'test' in raw_datasets:
        print(f"   Loaded {len(raw_datasets['test'])} test examples.")

    # 2. 加载模型和分词器
    print(f"\n2. Loading Tokenizer and Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 3. 数据预处理（tokenization）
    print("\n3. Preprocessing data...")

    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    print("   Preprocessing complete.")


    # 4. 设置训练参数与Trainer
    print("\n4. Setting up Training Arguments and Trainer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    model.to(device)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps=LOGGING_STEPS,

        save_strategy="epoch",

        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),

    )


    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],

        tokenizer=tokenizer,
        data_collator=data_collator,

    )

    # 5. 模型训练
    print("\n5. Starting Training...")
    train_result = trainer.train()
    print("   Training finished.")

    # 保存模型和分词器
    print("   Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   Model saved to {OUTPUT_DIR}")


    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    # 6. 测试集上生成预测结果
    if 'test' in tokenized_datasets:
        print("\n6. Generating predictions on the test set...")
        test_data_tokenized = tokenized_datasets["test"]

        predictions = trainer.predict(test_data_tokenized)

        decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

        cleaned_preds = [pred.strip() for pred in decoded_preds]

        # 7. 保存预测结果
        print(f"\n7. Saving predictions to {PREDICTIONS_FILE}...")
        with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
            for pred in cleaned_preds:
                f.write(pred + '\n')
        print("   Predictions saved.")

    else:
        print("\n6. No test data found. Skipping prediction generation.")


    print("\n--- Task Complete ---")


def compute_metrics_example(eval_preds):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]


    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]


    exact_matches = [1 if pred == label else 0 for pred, label in zip(decoded_preds, decoded_labels)]

    accuracy = sum(exact_matches) / len(exact_matches)


    f1_exact = f1_score([1]*len(decoded_labels), exact_matches)

    return {"exact_match_accuracy": accuracy, "f1_exact_placeholder": f1_exact}
