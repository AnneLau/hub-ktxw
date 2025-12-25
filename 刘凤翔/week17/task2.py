# 04_GPT4Rec.py
"""
GPT4Rec: 基于生成式语言模型的个性化推荐框架
实现论文《GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation》的核心思想
使用MovieLens数据集进行电影推荐
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')
import random
from tqdm import tqdm

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# 1. 数据加载与预处理
def load_and_preprocess_data(data_path='./data/'):
    """加载MovieLens数据集并进行预处理"""

    print("正在加载数据...")

    # 读取数据
    try:
        ratings = pd.read_csv(data_path + 'ratings.dat', sep='::', header=None, engine='python')
        movies = pd.read_csv(data_path + 'movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
    except FileNotFoundError:
        # 尝试不同的路径
        ratings = pd.read_csv('ratings.dat', sep='::', header=None, engine='python')
        movies = pd.read_csv('movies.dat', sep='::', header=None, engine='python', encoding='latin-1')

    # 设置列名
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]
    movies.columns = ["movie_id", "movie_title", "genres"]

    # 过滤评分数据，只保留高评分（4分及以上）
    ratings = ratings[ratings['rating'] >= 4]

    # 构建用户-电影序列
    print("构建用户序列...")
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings.sort_values(['user_id', 'timestamp'])

    # 按用户分组，获取电影序列
    user_sequences = ratings.groupby('user_id')['movie_id'].apply(list).reset_index()

    # 合并电影标题信息
    movie_dict = dict(zip(movies['movie_id'], movies['movie_title']))

    # 为每个用户的序列添加电影标题
    def get_titles(movie_ids):
        return [movie_dict.get(mid, f"Movie_{mid}") for mid in movie_ids]

    user_sequences['movie_titles'] = user_sequences['movie_id'].apply(get_titles)

    # 过滤掉序列太短的用户
    user_sequences = user_sequences[user_sequences['movie_id'].apply(len) >= 5]
    user_sequences = user_sequences[user_sequences['movie_id'].apply(len) <= 50]

    print(f"处理后用户数: {len(user_sequences)}")
    print(f"总电影数: {len(movies)}")

    return user_sequences, movies, ratings, movie_dict


# 2. 构建训练数据
def prepare_training_data(user_sequences, max_sequence_length=20):
    """为GPT-2模型准备训练数据"""

    print("准备训练数据...")

    # 提示模板（基于论文中的模板进行修改）
    PROMPT_TEMPLATE = """The user has previously watched these movies:
{history_titles}

Based on this, the user might search for movies about:"""

    train_samples = []

    for _, row in tqdm(user_sequences.iterrows(), total=len(user_sequences)):
        titles = row['movie_titles']
        movie_ids = row['movie_id']

        # 为每个序列位置创建训练样本
        for i in range(1, min(len(titles), max_sequence_length)):
            # 历史电影标题
            history = titles[:i]

            # 目标电影标题（作为查询）
            target = titles[i]

            # 构建输入文本
            history_text = "\n".join([f"- {title}" for title in history])
            input_text = PROMPT_TEMPLATE.format(history_titles=history_text)

            # 目标文本（查询）
            target_text = f" {target}"

            train_samples.append({
                'input_text': input_text,
                'target_text': target_text,
                'user_id': row['user_id'],
                'target_movie_id': movie_ids[i]
            })

    print(f"训练样本数: {len(train_samples)}")
    return train_samples


# 3. GPT-2模型微调类
class GPT4RecModel:
    """GPT4Rec模型类"""

    def __init__(self, model_name='gpt2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name

        # 初始化tokenizer和模型
        print(f"加载 {model_name} 模型...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # 添加填充token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.train()

        # 优化器参数
        self.optimizer = None
        self.scheduler = None

    def prepare_inputs(self, input_texts, target_texts=None, max_length=512):
        """准备模型输入"""

        # 如果提供了目标文本，则合并输入和目标
        if target_texts:
            combined_texts = [input_text + target_text for input_text, target_text in zip(input_texts, target_texts)]
        else:
            combined_texts = input_texts

        # Tokenize
        encodings = self.tokenizer(
            combined_texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # 创建标签（对于训练，我们需要预测目标部分）
        if target_texts:
            # 标记输入部分的位置
            input_encodings = self.tokenizer(
                input_texts,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )

            input_lengths = input_encodings['attention_mask'].sum(dim=1)

            # 创建标签，将输入部分设置为-100（忽略）
            labels = encodings['input_ids'].clone()
            for i, input_len in enumerate(input_lengths):
                labels[i, :input_len] = -100
        else:
            labels = None

        return {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device),
            'labels': labels.to(self.device) if labels is not None else None
        }

    def train(self, train_samples, batch_size=8, epochs=3, learning_rate=5e-5):
        """训练模型"""

        print("开始训练模型...")

        # 准备训练数据
        input_texts = [sample['input_text'] for sample in train_samples]
        target_texts = [sample['target_text'] for sample in train_samples]

        # 数据分批
        num_samples = len(train_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size

        # 优化器设置
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 学习率调度器（使用warmup）
        total_steps = num_batches * epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 打乱数据
            indices = list(range(num_samples))
            random.shuffle(indices)

            total_loss = 0
            progress_bar = tqdm(range(0, num_samples, batch_size))

            for start_idx in progress_bar:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # 获取批次数据
                batch_inputs = [input_texts[i] for i in batch_indices]
                batch_targets = [target_texts[i] for i in batch_indices]

                # 准备输入
                inputs = self.prepare_inputs(batch_inputs, batch_targets)

                # 前向传播
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )

                loss = outputs.loss
                total_loss += loss.item()

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # 更新进度条
                avg_loss = total_loss / (start_idx // batch_size + 1)
                progress_bar.set_description(f"Loss: {avg_loss:.4f}")

        print(f"训练完成，最终平均损失: {total_loss / num_batches:.4f}")

    def generate_queries(self, user_history, num_queries=5, beam_size=5, max_length=50):
        """使用beam search生成多个查询"""

        self.model.eval()

        # 构建输入文本
        history_text = "\n".join([f"- {title}" for title in user_history])
        input_text = f"""The user has previously watched these movies:
{history_text}

Based on this, the user might search for movies about:"""

        # Tokenize输入
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Beam search生成
        with torch.no_grad():
            # 使用多样化的beam search
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                num_return_sequences=num_queries,
                num_beams=beam_size,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )

        # 解码生成的查询
        generated_queries = []
        for output in outputs:
            # 只获取生成的部分（去掉输入）
            generated = output[len(inputs['input_ids'][0]):]
            query = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_queries.append(query.strip())

        return generated_queries


# 4. BM25检索器类
class BM25Retriever:
    """BM25检索器类"""

    def __init__(self, movie_data):
        """初始化BM25检索器"""

        print("初始化BM25检索器...")

        self.movie_data = movie_data

        # 准备文档（电影标题）
        self.documents = []
        self.movie_ids = []

        for _, row in movie_data.iterrows():
            movie_id = row['movie_id']
            title = row['movie_title']
            genres = row['genres']

            # 将电影标题和类型合并作为文档
            doc = f"{title} {genres}".lower()
            self.documents.append(doc)
            self.movie_ids.append(movie_id)

        # Tokenize文档
        tokenized_docs = [self.tokenize(doc) for doc in self.documents]

        # 初始化BM25
        self.bm25 = BM25Okapi(tokenized_docs)

    def tokenize(self, text):
        """简单的tokenization"""
        return text.split()

    def search(self, query, top_k=10, exclude_movies=None):
        """搜索与查询最相关的电影"""

        if exclude_movies is None:
            exclude_movies = set()

        # Tokenize查询
        tokenized_query = self.tokenize(query.lower())

        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top-k结果（排除已看过的电影）
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            movie_id = self.movie_ids[idx]

            # 排除已看过的电影
            if movie_id in exclude_movies:
                continue

            results.append({
                'movie_id': movie_id,
                'title': self.movie_data[self.movie_data['movie_id'] == movie_id]['movie_title'].iloc[0],
                'score': scores[idx]
            })

            if len(results) >= top_k:
                break

        return results


# 5. GPT4Rec推荐系统类
class GPT4RecSystem:
    """完整的GPT4Rec推荐系统"""

    def __init__(self, model_name='gpt2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        # 初始化组件
        self.gpt_model = GPT4RecModel(model_name, self.device)
        self.bm25_retriever = None
        self.movie_dict = None

    def train(self, train_samples):
        """训练GPT-2模型"""
        self.gpt_model.train(train_samples)

    def setup_retriever(self, movie_data):
        """设置检索器"""
        self.bm25_retriever = BM25Retriever(movie_data)
        self.movie_data = movie_data

    def recommend(self, user_history, k=10, num_queries=5):
        """为用户生成推荐"""

        if self.bm25_retriever is None:
            raise ValueError("请先调用setup_retriever()设置检索器")

        # 生成查询
        print(f"为用户生成 {num_queries} 个查询...")
        queries = self.gpt_model.generate_queries(user_history, num_queries=num_queries)

        print("生成的查询:")
        for i, query in enumerate(queries):
            print(f"  查询 {i + 1}: {query}")

        # 排除用户已经看过的电影
        exclude_movies = set()
        # 注意：这里需要电影ID，但我们只有标题，所以跳过这一步

        # 为每个查询检索电影
        all_results = []
        seen_movies = set()

        for query in queries:
            results = self.bm25_retriever.search(query, top_k=k // num_queries + 2)

            for result in results:
                movie_id = result['movie_id']

                # 去重
                if movie_id not in seen_movies:
                    seen_movies.add(movie_id)
                    all_results.append(result)

        # 按分数排序
        all_results.sort(key=lambda x: x['score'], reverse=True)

        # 返回top-k结果
        return all_results[:k]

    def evaluate(self, test_users, test_targets, k=10, num_queries=5):
        """评估推荐系统"""

        print(f"\n评估推荐系统 (K={k})...")

        hits = 0
        total_users = len(test_users)

        for user_id, (history, target_movie_id) in tqdm(zip(test_users, test_targets.items()), total=total_users):
            # 生成推荐
            recommendations = self.recommend(history, k=k, num_queries=num_queries)

            # 检查目标电影是否在推荐中
            recommended_movie_ids = [rec['movie_id'] for rec in recommendations]

            if target_movie_id in recommended_movie_ids:
                hits += 1

        recall_at_k = hits / total_users
        print(f"Recall@{k}: {recall_at_k:.4f}")

        return recall_at_k


# 6. 主函数
def main():
    """主函数"""

    print("=" * 60)
    print("GPT4Rec: 基于生成式语言模型的个性化推荐系统")
    print("=" * 60)

    # 步骤1: 加载和预处理数据
    print("\n[步骤1] 数据加载与预处理")
    user_sequences, movies, ratings, movie_dict = load_and_preprocess_data()

    # 步骤2: 准备训练数据
    print("\n[步骤2] 准备训练数据")
    train_samples = prepare_training_data(user_sequences)

    # 划分训练集和测试集
    train_samples, test_samples = train_test_split(
        train_samples, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {len(train_samples)}")
    print(f"测试集大小: {len(test_samples)}")

    # 步骤3: 初始化GPT4Rec系统
    print("\n[步骤3] 初始化GPT4Rec系统")

    # 使用小模型以节省内存（如果显存不足）
    model_name = 'distilgpt2'  # 使用distilgpt2作为轻量级替代
    gpt4rec = GPT4RecSystem(model_name=model_name)

    # 设置检索器
    gpt4rec.setup_retriever(movies)

    # 步骤4: 训练模型（如果显存不足，可以跳过训练，直接加载预训练模型）
    print("\n[步骤4] 训练GPT-2模型")
    try:
        gpt4rec.train(train_samples[:500])  # 使用少量样本进行演示
        print("模型训练完成！")
    except RuntimeError as e:
        print(f"训练时出现错误（可能是显存不足）: {e}")
        print("跳过训练，使用预训练模型进行生成...")

    # 步骤5: 测试推荐
    print("\n[步骤5] 测试推荐系统")

    # 准备测试数据
    test_users = {}
    test_targets = {}

    for sample in test_samples[:20]:  # 使用少量测试样本
        user_id = sample['user_id']
        input_text = sample['input_text']

        # 从输入文本中提取历史电影
        lines = input_text.split('\n')
        history_lines = [line for line in lines if line.startswith('- ')]
        history = [line[2:] for line in history_lines]

        test_users[user_id] = history
        test_targets[user_id] = sample['target_movie_id']

    # 为第一个用户生成推荐
    print("\n示例推荐:")
    first_user_id = list(test_users.keys())[0]
    first_user_history = test_users[first_user_id]
    first_user_target = test_targets[first_user_id]

    print(f"用户ID: {first_user_id}")
    print(f"历史观看电影:")
    for i, movie in enumerate(first_user_history[:10]):
        print(f"  {i + 1}. {movie}")

    print(f"\n目标电影: {movie_dict.get(first_user_target, f'Movie_{first_user_target}')}")

    # 生成推荐
    recommendations = gpt4rec.recommend(first_user_history, k=10, num_queries=5)

    print(f"\n推荐的电影:")
    for i, rec in enumerate(recommendations):
        print(f"  {i + 1}. {rec['title']} (分数: {rec['score']:.4f})")

        # 检查是否是目标电影
        if rec['movie_id'] == first_user_target:
            print("    ^ 命中目标电影!")

    # 步骤6: 评估（可选，需要更多计算资源）
    print("\n[步骤6] 评估推荐系统")

    try:
        recall_at_10 = gpt4rec.evaluate(
            list(test_users.keys())[:10],  # 使用前10个用户进行评估
            test_targets,
            k=10,
            num_queries=5
        )
        print(f"Recall@10: {recall_at_10:.4f}")
    except Exception as e:
        print(f"评估时出错: {e}")
        print("跳过评估...")

    print("\n" + "=" * 60)
    print("GPT4Rec系统运行完成!")
    print("=" * 60)

    return gpt4rec


# 7. 交互式推荐函数
def interactive_recommendation(gpt4rec, movie_dict):
    """交互式推荐功能"""

    print("\n交互式电影推荐")
    print("输入你喜欢的电影（用逗号分隔），输入'quit'退出")

    while True:
        user_input = input("\n请输入电影名称: ").strip()

        if user_input.lower() == 'quit':
            break

        if not user_input:
            continue

        # 解析输入的电影
        movie_titles = [title.strip() for title in user_input.split(',')]

        print(f"\n根据你的电影喜好: {', '.join(movie_titles)}")
        print("正在生成推荐...")

        # 生成推荐
        recommendations = gpt4rec.recommend(movie_titles, k=10, num_queries=5)

        print(f"\n为你推荐的电影:")
        for i, rec in enumerate(recommendations):
            print(f"  {i + 1}. {rec['title']}")


# 运行主函数
if __name__ == "__main__":

    # 检查数据文件是否存在
    import os

    if not os.path.exists('ratings.dat') or not os.path.exists('movies.dat'):
        print("错误: 未找到MovieLens数据文件!")
        print("请确保以下文件在当前目录中:")
        print("  - ratings.dat (MovieLens 1M数据集)")
        print("  - movies.dat (MovieLens 1M数据集)")
        print("\n你可以从 https://grouplens.org/datasets/movielens/1m/ 下载")

        # 创建示例数据（用于演示）
        print("\n创建示例数据用于演示...")
        create_sample_data()

    # 运行主程序
    gpt4rec_system = main()

    # 运行交互式推荐
    try:
        interactive_recommendation(gpt4rec_system, movie_dict={})
    except NameError:
        print("无法运行交互式推荐，需要完整的movie_dict")