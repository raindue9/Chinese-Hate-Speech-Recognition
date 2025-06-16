import numpy as np
import pandas as pd
import torch
import json
import re
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2
from sklearn.metrics import accuracy_score, f1_score
from torchcrf import CRF

# --- 配置 ---
TRAIN_FILE = 'train.json'
TEST_FILE = 'test1.json'
OUTPUT_FILE = 'predictions_seqlabel.txt'

PRETRAINED_MODEL = 'hfl/chinese-roberta-wwm-ext'


LABELER_MODEL_SAVE_PATH = 'roberta_hate_labeler.pt'
CLASSIFIER_MODEL_SAVE_PATH = 'roberta_hate_classifier.pt'


MAX_LENGTH = 128
LABELER_BATCH_SIZE = 16
CLASSIFIER_BATCH_SIZE = 16
LABELER_EPOCHS = 5
CLASSIFIER_EPOCHS = 5
LABELER_LR = 3e-5
CLASSIFIER_LR = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

# --- 设置随机种子 ---
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)

# 定义BIO标注标签，用于序列标注任务
tag2id = {'O': 0, 'B-TGT': 1, 'I-TGT': 2, 'B-ARG': 3, 'I-ARG': 4}
id2tag = {v: k for k, v in tag2id.items()}
NUM_BIO_LABELS = len(tag2id)

# 解析模型输出的字符串
def parse_output(output):

    quadruples = []
    if not isinstance(output, str): return quadruples


    parts = output.strip().replace(' [END]', '').split(' [SEP] ')
    for part in parts:
        elements = part.split(' | ')
        if len(elements) == 4:
            quadruples.append({
                'Target': elements[0].strip(),
                'Argument': elements[1].strip(),
                'Targeted Group': elements[2].strip(),
                'Hateful': elements[3].strip()
            })
    return quadruples

# 在一个列表中查找子列表的起止索引，用于匹配Token序列
def find_sublist_indices(main_list, sub_list):

    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i+len(sub_list)] == sub_list:
            return i, i + len(sub_list) - 1
    return -1, -1

# 生成BIO标签
def create_bio_tags(text, target, argument, tokenizer, max_length):

    encoding = tokenizer(text, max_length=max_length, truncation=True, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    tags = ['O'] * len(tokens)


    def tag_span(span_text, b_tag, i_tag):
        if not span_text or span_text.lower() == 'null':
            return

        span_tokens = tokenizer.tokenize(span_text)
        if not span_tokens: return


        start_idx, end_idx = find_sublist_indices(tokens, span_tokens)


        if start_idx == -1:
            span_char_start = text.find(span_text)
            if span_char_start != -1:
                span_char_end = span_char_start + len(span_text)
                token_start_idx, token_end_idx = -1, -1
                for idx, (offset_start, offset_end) in enumerate(offsets):

                    if offset_start == 0 and offset_end == 0: continue
                    if offset_start >= span_char_start and offset_end <= span_char_end:
                        if token_start_idx == -1:
                            token_start_idx = idx
                        token_end_idx = idx

                if token_start_idx != -1 and token_end_idx != -1:
                    start_idx = token_start_idx
                    end_idx = token_end_idx



        if start_idx != -1:
            tags[start_idx] = b_tag
            for i in range(start_idx + 1, end_idx + 1):

                if tags[i] == 'O':
                    tags[i] = i_tag


    tag_span(target, 'B-TGT', 'I-TGT')
    tag_span(argument, 'B-ARG', 'I-ARG')


    tag_ids = []
    for i, tag in enumerate(tags):

        offset = offsets[i]
        if offset == (0, 0) or encoding['input_ids'][i] == tokenizer.pad_token_id:
            tag_ids.append(-100)
        else:
            tag_ids.append(tag2id[tag])

    return tag_ids

# 将BIO标签解码
def decode_tags_to_spans(tokens, tags, tokenizer):

    spans = {'Target': [], 'Argument': []}
    current_span_type = None
    current_span_tokens = []


    min_len = min(len(tokens), len(tags))
    tokens = tokens[:min_len]
    tags = tags[:min_len]

    for token, tag in zip(tokens, tags):

        if tag.startswith('B-'):

            if current_span_tokens:
                span_text = tokenizer.convert_tokens_to_string(current_span_tokens).replace(' ', '')
                if span_text:
                    if current_span_type == 'TGT': spans['Target'].append(span_text)
                    elif current_span_type == 'ARG': spans['Argument'].append(span_text)


            current_span_type = tag.split('-')[1]
            current_span_tokens = [token]

        elif tag.startswith('I-'):
            span_type = tag.split('-')[1]

            if current_span_type == span_type and current_span_tokens:
                current_span_tokens.append(token)
            else:
                if current_span_tokens:
                    span_text = tokenizer.convert_tokens_to_string(current_span_tokens).replace(' ', '')
                    if span_text:
                        if current_span_type == 'TGT': spans['Target'].append(span_text)
                        elif current_span_type == 'ARG': spans['Argument'].append(span_text)
                current_span_type = None
                current_span_tokens = []

        else:
            if current_span_tokens:
                span_text = tokenizer.convert_tokens_to_string(current_span_tokens).replace(' ', '')
                if span_text:
                    if current_span_type == 'TGT': spans['Target'].append(span_text)
                    elif current_span_type == 'ARG': spans['Argument'].append(span_text)
            current_span_type = None
            current_span_tokens = []


    if current_span_tokens:
        span_text = tokenizer.convert_tokens_to_string(current_span_tokens).replace(' ', '')
        if span_text:
            if current_span_type == 'TGT': spans['Target'].append(span_text)
            elif current_span_type == 'ARG': spans['Argument'].append(span_text)


    final_pairs = []
    targets = spans['Target'] if spans['Target'] else ["NULL"]
    arguments = spans['Argument']

    if not arguments:
        for tgt in targets:
            final_pairs.append((tgt, "NULL"))
            if len(final_pairs) >= 3: break
    elif not spans['Target']:

        for arg in arguments:
            final_pairs.append(("NULL", arg))
            if len(final_pairs) >= 3: break
    else:

        num_pairs = max(len(targets), len(arguments))
        for i in range(num_pairs):
            tgt = targets[i] if i < len(targets) else targets[-1]
            arg = arguments[i] if i < len(arguments) else arguments[-1]
            final_pairs.append((tgt, arg))
            if len(final_pairs) >= 3: break


    if not final_pairs:

        final_pairs.append(("NULL", "NULL"))


    return final_pairs[:3]


# 1. 初始化数据与分词器
class SequenceLabelingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']
        target = item['Target']
        argument = item['Argument']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )


        labels = create_bio_tags(text, target, argument, self.tokenizer, self.max_length)


        labels = labels + [-100] * (self.max_length - len(labels))

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# 2. 分类模型
class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, group_encoder, hateful_encoder):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_encoder = group_encoder     # 编码“Targeted Group”标签
        self.hateful_encoder = hateful_encoder # 编码“Hateful”标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']
        target = item['Target']
        argument = item['Argument']
        group = item['Targeted Group']
        hateful = item['Hateful']


        # 构造输入文本，拼接目标、论点和上下文信息
        input_text = f"[CLS] TGT: {target} [SEP] ARG: {argument} [SEP] CTX: {content} [SEP]"

        encoding = self.tokenizer(
            input_text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'group_labels': torch.tensor(self.group_encoder.transform([group])[0], dtype=torch.long),
            'hateful_labels': torch.tensor(self.hateful_encoder.transform([hateful])[0], dtype=torch.long)
        }


# --- Models ---

# 1. 用于序列标注的模型（BIO标签）
def get_labeler_model():
    model = BertForTokenClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_BIO_LABELS
    )
    return model

# 2. 多任务分类模型（Group + Hateful）
class MultiTaskHateClassifier(torch.nn.Module):
    def __init__(self, num_group_labels, num_hateful_labels):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL,
            num_labels=num_group_labels  # 初始值，用于替换
        )

        bert_config = self.bert.config
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        # 预测Targeted Group
        self.group_classifier = torch.nn.Linear(bert_config.hidden_size, num_group_labels)
        # 预测Hateful标签
        self.hateful_classifier = torch.nn.Linear(bert_config.hidden_size, num_hateful_labels)

    def forward(self, input_ids, attention_mask, group_labels=None, hateful_labels=None):

        outputs = self.bert.bert( # 使用底层BERT
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        # 获取[CLS]
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        group_logits = self.group_classifier(pooled_output)
        hateful_logits = self.hateful_classifier(pooled_output)

        loss = None
        if group_labels is not None and hateful_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            group_loss = loss_fct(group_logits.view(-1, self.group_classifier.out_features), group_labels.view(-1))
            hateful_loss = loss_fct(hateful_logits.view(-1, self.hateful_classifier.out_features), hateful_labels.view(-1))
            loss = group_loss + hateful_loss

        return {
            'loss': loss,
            'group_logits': group_logits,
            'hateful_logits': hateful_logits
        }

# 模型训练与评估

def train_labeler_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training Labeler"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

# 评估序列标注器
def evaluate_labeler(model, dataloader, device, id2tag):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Labeler"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)

            # 对齐预测和真实标签，只保留非-100项
            for i in range(labels.shape[0]):
                pred_seq = []
                label_seq = []
                for pred_id, label_id in zip(predictions[i].tolist(), labels[i].tolist()):
                    if label_id != -100:
                        pred_seq.append(id2tag[pred_id])
                        label_seq.append(id2tag[label_id])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    avg_loss = total_loss / len(dataloader)

    report = seqeval_classification_report(all_labels, all_preds, mode='strict', scheme=IOB2, output_dict=True)
    f1 = report['micro avg']['f1-score']
    print(f"Labeler Eval Loss: {avg_loss:.4f}")
    print(seqeval_classification_report(all_labels, all_preds, mode='strict', scheme=IOB2))
    return avg_loss, f1

# 训练多任务分类器
def train_classifier_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training Classifier"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        group_labels = batch['group_labels'].to(device)
        hateful_labels = batch['hateful_labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            group_labels=group_labels,
            hateful_labels=hateful_labels
        )

        loss = outputs['loss']
        if loss is None: continue
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


# 评估多任务分类器
def evaluate_classifier(model, dataloader, device, group_encoder, hateful_encoder):
    model.eval()
    total_loss = 0
    all_group_preds, all_group_labels = [], []
    all_hateful_preds, all_hateful_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Classifier"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            group_labels = batch['group_labels'].to(device)
            hateful_labels = batch['hateful_labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                group_labels=group_labels,
                hateful_labels=hateful_labels
            )

            loss = outputs['loss']
            total_loss += loss.item()

            group_preds = torch.argmax(outputs['group_logits'], dim=1).cpu().numpy()
            hateful_preds = torch.argmax(outputs['hateful_logits'], dim=1).cpu().numpy()

            all_group_preds.extend(group_preds)
            all_group_labels.extend(group_labels.cpu().numpy())
            all_hateful_preds.extend(hateful_preds)
            all_hateful_labels.extend(hateful_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    group_accuracy = accuracy_score(all_group_labels, all_group_preds)
    group_f1 = f1_score(all_group_labels, all_group_preds, average='macro')
    hateful_accuracy = accuracy_score(all_hateful_labels, all_hateful_preds)
    hateful_f1 = f1_score(all_hateful_labels, all_hateful_preds, average='macro')

    print(f"Classifier Eval Loss: {avg_loss:.4f}")
    print(f"Group Acc: {group_accuracy:.4f}, F1: {group_f1:.4f}")
    print(f"Hateful Acc: {hateful_accuracy:.4f}, F1: {hateful_f1:.4f}")

    return avg_loss, (group_f1 + hateful_f1) / 2 # Return average F1


if __name__ == "__main__":
    # 1. 加载数据
    print("Loading data...")
    df_train_raw = pd.read_json(TRAIN_FILE)
    df_test_raw = pd.read_json(TEST_FILE)

    # 2. 准备Labeler和Classifier所需的数据（用于提取和分类）
    print("Preparing data...")
    labeler_data = []
    classifier_data = []
    all_groups = set()
    all_hatefuls = set()

    for _, row in df_train_raw.iterrows():
        content = row['content']
        quadruples = parse_output(row.get('output', ''))

        if not quadruples:
            continue

        for quad in quadruples:
            labeler_data.append({
                'content': content,
                'Target': quad['Target'],
                'Argument': quad['Argument']
            })
            classifier_data.append({
                'content': content,
                'Target': quad['Target'],
                'Argument': quad['Argument'],
                'Targeted Group': quad['Targeted Group'],
                'Hateful': quad['Hateful']
            })
            all_groups.add(quad['Targeted Group'])
            all_hatefuls.add(quad['Hateful'])

    print(f"Processed {len(labeler_data)} samples for labeler.")
    print(f"Processed {len(classifier_data)} samples for classifier.")
    print(f"Target Groups: {all_groups}")
    print(f"Hateful Statuses: {all_hatefuls}")

    group_encoder = LabelEncoder().fit(list(all_groups))
    hateful_encoder = LabelEncoder().fit(list(all_hatefuls))
    NUM_GROUP_LABELS = len(group_encoder.classes_)
    NUM_HATEFUL_LABELS = len(hateful_encoder.classes_)

    # 3. 跳过训练，加载已经训练好的模型
    print("\n--- Loading Trained Models ---")
    labeler_model = get_labeler_model().to(device)
    labeler_model.load_state_dict(torch.load(LABELER_MODEL_SAVE_PATH, map_location=device))
    labeler_model.eval()

    classifier_model = MultiTaskHateClassifier(NUM_GROUP_LABELS, NUM_HATEFUL_LABELS).to(device)
    classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_SAVE_PATH, map_location=device))
    classifier_model.eval()


    # 4. 在测试集上进行预测
    print("\n--- Predicting on Test Set ---")
    test_predictions_formatted = []

    with torch.no_grad():
        for _, row in tqdm(df_test_raw.iterrows(), total=len(df_test_raw), desc="Predicting Test"):
            text = row['content']
            item_id = row['id']

            # 第一步：使用Labeler模型抽取Target/Argument对
            encoding = tokenizer(text, max_length=MAX_LENGTH, truncation=True, return_tensors='pt', return_offsets_mapping=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            offsets = encoding['offset_mapping'].squeeze().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

            labeler_outputs = labeler_model(input_ids, attention_mask=attention_mask)
            predicted_tag_ids = torch.argmax(labeler_outputs.logits, dim=-1).squeeze().cpu().tolist()

            # 去除特殊token对应的预测
            valid_tokens = []
            valid_predicted_tags = []
            for tok, offset, tag_id in zip(tokens, offsets, predicted_tag_ids):
                if offset != (0, 0):
                    valid_tokens.append(tok)
                    valid_predicted_tags.append(id2tag[tag_id])



            extracted_pairs = decode_tags_to_spans(valid_tokens, valid_predicted_tags,tokenizer)


            # 第二步：对抽取到的每个(Target, Argument)进行分类
            quadruples_for_item = []
            if not extracted_pairs:
                extracted_pairs = [("NULL", text)]

            for target, argument in extracted_pairs:

                clf_input_text = f"[CLS] TGT: {target} [SEP] ARG: {argument} [SEP] CTX: {text} [SEP]"
                clf_encoding = tokenizer(
                    clf_input_text,
                    add_special_tokens=False,
                    max_length=MAX_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                clf_input_ids = clf_encoding['input_ids'].to(device)
                clf_attention_mask = clf_encoding['attention_mask'].to(device)


                clf_outputs = classifier_model(clf_input_ids, attention_mask=clf_attention_mask)
                group_pred_id = torch.argmax(clf_outputs['group_logits'], dim=1).item()
                hateful_pred_id = torch.argmax(clf_outputs['hateful_logits'], dim=1).item()


                predicted_group = group_encoder.inverse_transform([group_pred_id])[0]
                predicted_hateful = hateful_encoder.inverse_transform([hateful_pred_id])[0]

                # 后处理规则（可选）：根据敏感词修正分类结
                combined_check_text = f"{target} {argument} {text}"
                if any(term in combined_check_text.lower() for term in ['nigger', 'negro', '黑鬼']):
                    predicted_group = "Racism"
                    predicted_hateful = "hate"

                quadruples_for_item.append(f"{target} | {argument} | {predicted_group} | {predicted_hateful}")

            # 格式化输出结果
            if not quadruples_for_item:
                final_output_str = "NULL | {} | non-hate | non-hate [END]".format(text)
            else:
                final_output_str = " [SEP] ".join(quadruples_for_item) + " [END]"

            test_predictions_formatted.append(final_output_str)

    # 5. 保存预测结果
    print(f"\nSaving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for pred in test_predictions_formatted:
            f.write(pred + '\n')

    print("Predictions saved.")


    # 显示前5条预测示例
    print("\nTest Prediction Examples:")
    for i in range(min(5, len(df_test_raw))):
        print(f"ID: {df_test_raw.iloc[i]['id']}")
        print(f"Content: {df_test_raw.iloc[i]['content']}")
        print(f"Prediction: {test_predictions_formatted[i]}")
        print("-" * 20)