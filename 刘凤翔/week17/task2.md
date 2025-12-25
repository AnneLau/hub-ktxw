
## 运行说明
## 创建示例数据的函数

```python
def create_sample_data():
    """创建示例数据（如果真实数据不可用）"""
    
    import pandas as pd
    
    # 创建示例电影数据
    movies_data = {
        'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'movie_title': [
            'Toy Story (1995)',
            'Jumanji (1995)',
            'Grumpier Old Men (1995)',
            'Waiting to Exhale (1995)',
            'Father of the Bride Part II (1995)',
            'Heat (1995)',
            'Sabrina (1995)',
            'Tom and Huck (1995)',
            'Sudden Death (1995)',
            'GoldenEye (1995)'
        ],
        'genres': [
            'Animation|Children|Comedy',
            'Adventure|Children|Fantasy',
            'Comedy|Romance',
            'Comedy|Drama|Romance',
            'Comedy',
            'Action|Crime|Thriller',
            'Comedy|Romance',
            'Adventure|Children',
            'Action',
            'Action|Adventure|Thriller'
        ]
    }
    
    movies_df = pd.DataFrame(movies_data)
    movies_df.to_csv('movies.dat', sep='::', header=False, index=False)
    
    # 创建示例评分数据
    ratings_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'movie_id': [1, 2, 3, 2, 4, 5, 1, 6, 7, 8, 9, 10],
        'rating': [5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4],
        'timestamp': [978300760, 978302109, 978301968, 
                      978300275, 978302268, 978301752,
                      978300761, 978302110, 978301969,
                      978300276, 978302269, 978301753]
    }
    
    ratings_df = pd.DataFrame(ratings_data)
    ratings_df.to_csv('ratings.dat', sep='::', header=False, index=False)
    
    print("示例数据已创建！")
    print("movies.dat 和 ratings.dat 文件已生成在当前目录")
```

## 安装依赖

创建一个 `requirements.txt` 文件：

```txt
torch>=1.9.0
transformers>=4.18.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
rank-bm25>=0.2.1
nltk>=3.7.0
tqdm>=4.64.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 运行说明

1. **数据准备**：
   - 从 https://grouplens.org/datasets/movielens/1m/ 下载MovieLens 1M数据集
   - 将 `ratings.dat` 和 `movies.dat` 放在当前目录
   - 或者运行代码，它会自动创建示例数据

2. **运行程序**：
```bash
python 04_GPT4Rec.py
```

3. **主要功能**：
   - 加载和预处理MovieLens数据
   - 微调GPT-2模型生成用户查询
   - 使用BM25检索相关电影
   - 生成个性化推荐
   - 交互式推荐界面

4. **注意事项**：
   - 如果显存不足，可以减少训练样本数量
   - 可以使用 `distilgpt2` 替代 `gpt2` 以减少内存使用
   - 可以调整 `batch_size` 和 `epochs` 参数

这个实现完整复现了GPT4Rec论文的核心思想，包括：
1. **查询生成**：使用GPT-2根据用户历史生成搜索查询
2. **多查询生成**：使用beam search生成多个查询
3. **检索阶段**：使用BM25检索相关电影
4. **结果合并**：合并多个查询的检索结果

代码具有模块化设计，易于扩展和调整。