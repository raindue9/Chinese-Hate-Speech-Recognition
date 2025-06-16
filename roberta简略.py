import torch
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification



# 核心配置
MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
MAX_LENGTH = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


# BIO（Begin-Inside-Outside）
# B-TGT	Target 开始位置（Begin）
# I-TGT	Target 中间/后续 token（Inside）
# B-ARG	Argument 开始位置
# I-ARG	Argument 中间/后续 token
# O	Outside，表示不是目标的 token

# 模型BertForTokenClassification 对 每一个输入 token 输出一个大小为 num_labels 的向量，然后通过 softmax 得到每个标签的概率分布
# 如： tokens: ["某些", "同性", "恋者", "传播", "艾滋病", ...]
#     labels: ["O", "B-TGT", "I-TGT", "B-ARG", "I-ARG", ...]
# 处理规则：
# B-TGT/I-TGT：合并连续的 TGT 标签，得到完整的 Target 文本片段。
# B-ARG/I-ARG：同理合并 Argument 文本片段
# 然后对 Target-Argument 对做分类预测  Group Hateful
# --- 核心函数：BIO标签解码 ---
def decode_tags_to_spans(tokens, tags):
    # 将BIO标签转换为(Target, Argument)对
    spans = {'Target': [], 'Argument': []}
    current_tag = None
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_tokens:  # 保存上一个span
                spans[current_tag].append(''.join(current_tokens).replace('##', ''))
            current_tag = 'Target' if 'TGT' in tag else 'Argument'
            current_tokens = [token]
        elif tag.startswith('I-'):
            current_tokens.append(token)
        else:
            if current_tokens:
                spans[current_tag].append(''.join(current_tokens).replace('##', ''))
            current_tag = None
            current_tokens = []

    # 处理最后一个span
    if current_tokens:
        spans[current_tag].append(''.join(current_tokens).replace('##', ''))

    return spans['Target'], spans['Argument']


# --- 四元组生成流程 ---
def predict_quadruples(text, labeler_model, classifier_model):

    # Step1: 序列标注提取Target/Argument
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = labeler_model(**encoding).logits
    tags = logits.argmax(-1).squeeze().cpu().numpy()

    # 解码BIO标签
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    valid_tokens = [t for t, idx in zip(tokens, encoding['input_ids'][0]) if idx not in tokenizer.all_special_ids]
    valid_tags = [id2tag[t] for t, idx in zip(tags, encoding['input_ids'][0]) if idx not in tokenizer.all_special_ids]
    targets, arguments = decode_tags_to_spans(valid_tokens, valid_tags)

    # Step2: 分类预测Group和Hateful
    quadruples = []
    for target in targets[:3]:  # 最多取3个
        for arg in arguments[:3]:
            # 拼接分类输入
            inputs = tokenizer(f"TGT:{target} ARG:{arg} CTX:{text}",
                               truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)

            # 双分类预测
            with torch.no_grad():
                outputs = classifier_model(**inputs)
            group = group_labels[outputs.group_logits.argmax().item()]
            hateful = hate_labels[outputs.hateful_logits.argmax().item()]

            quadruples.append(f"{target} | {arg} | {group} | {hateful}")

    return ' [SEP] '.join(quadruples) + ' [END]' if quadruples else 'NULL | NULL | non-hate | non-hate [END]'


# --- 演示使用 ---
if __name__ == "__main__":
    # 加载预训练模型
    labeler_model = BertForTokenClassification.from_pretrained(MODEL_NAME).to(device)
    classifier_model = BertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    # 示例预测
    test_text = "某些同性恋者传播艾滋病，应该被隔离！"
    # 同性恋者 | 传播艾滋病 | LGBTQ | hate [SEP] 艾滋病 | 隔离 | others | hate [END]
    print(predict_quadruples(test_text, labeler_model, classifier_model))