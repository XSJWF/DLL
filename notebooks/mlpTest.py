import os
# import sys

import jieba
import numpy as np
import torch
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# sys.path.insert(0, os.getcwd())
# os.add_dll_directory(r"D:\Anaconda3\envs\DL\Lib\site-packages\libtorch\lib")
#
# # 导入 C++ 中的 MLP 模型
# from Release.cpp_mlp import MLP  # 这里是假设通过 pybind11 绑定的 C++ 类
from torch.utils.cpp_extension import load
cpp_mlp = load(name='cpp_mlp',
               sources=['cpp_mlp.cpp'],
               verbose=True)

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 加载数据集（与原代码相同）
def load_data(data_dir, num_samples_per_class=1000):
    categories = {
        '体育': 'sports',
        '财经': 'finance',
        '房产': 'real_estate',
        '家居': 'home',
        '教育': 'education',
        '科技': 'technology',
        '时尚': 'fashion',
        '时政': 'politics',
        '游戏': 'games',
        '娱乐': 'entertainment'
    }

    texts = []
    labels = []
    label_map = {}
    label_idx = 0

    print("加载中文文本数据...")
    for category, folder in categories.items():
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            continue

        label_map[label_idx] = category
        files = os.listdir(category_dir)[:num_samples_per_class]

        for file in files:
            file_path = os.path.join(category_dir, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    labels.append(label_idx)

        print(f"类别 '{category}' 加载完成, 样本数: {len(files)}")
        label_idx += 1
    return texts, labels, label_map


# 加载停用词
def stop_words():
    with open('../data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.rstrip('\n') for line in f]
    return stopwords


# 中文文本预处理
def preprocess_chinese_text(text, stopwords):
    words = jieba.cut(text)
    words = [word for word in words if word not in stopwords and len(word) >= 1]
    return ' '.join(words)


# 创建数据集类
class ChineseTextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        labels: torch.Tensor
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total_acc += correct
        total_count += labels.size(0)

    return total_loss / len(loader), total_acc / total_count


# 评估函数
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            labels: torch.Tensor

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total_acc += correct
            total_count += labels.size(0)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / len(loader), total_acc / total_count, all_preds, all_labels


# 主程序
def main():
    data_dir = r'E:\THUCNews'
    texts, labels, label_map = load_data(data_dir, num_samples_per_class=500)
    stopwords = stop_words()

    # 预处理文本
    processed_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        processed = preprocess_chinese_text(text, stopwords)
        if processed.strip():
            processed_texts.append(processed)
            valid_indices.append(i)

    labels = [labels[i] for i in valid_indices]

    # 特征提取
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(processed_texts).toarray()
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 创建数据集和数据加载器
    train_dataset = ChineseTextDataset(X_train, y_train)
    test_dataset = ChineseTextDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化 C++ MLP 模型
    input_dim = X_train.shape[1]
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = len(label_map)
    print(1)
    model = cpp_mlp.MLP(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
    print(2)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(15):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)

        print(f'Epoch {epoch + 1}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc * 100:.2f}%')
        print(f'Epoch {epoch + 1}, 测试损失: {test_loss:.4f}, 测试准确率: {test_acc * 100:.2f}%')

    # 测试
    print("测试模型中...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    # 修正：构造字符串列表作为 target_names
    target_names = [label_map[i] for i in range(len(label_map))]
    print(classification_report(test_labels, test_preds, target_names=target_names))
    print(confusion_matrix(test_labels, test_preds))


if __name__ == '__main__':
    main()
