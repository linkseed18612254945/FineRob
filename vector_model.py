from FlagEmbedding import BGEM3FlagModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
vector_model = BGEM3FlagModel('/public/llms/bge-m3', use_fp16=True)

def get_m3e_vectors(text):
    if isinstance(text, str):
        sentences = [text]
    else:
        sentences = text
    embeddings = vector_model.encode(sentences, batch_size=1, max_length=1024)['dense_vecs']
    return np.mean(embeddings, axis=0)


def find_top_k_similar(hist_vectors, curr_vector, k):
    """
    输入：
    - hist_vectors: 历史文本的向量矩阵 (numpy array, shape: [num_texts, vector_dim])
    - curr_vector: 当前文本的向量 (numpy array, shape: [vector_dim])
    - k: 返回最相似的前k个文本的序号 (int)

    输出：
    - top_k_indices: 最相似的前k个文本的序号 (list of int)
    """
    # 计算余弦相似度
    similarities = cosine_similarity(hist_vectors, curr_vector.reshape(1, -1)).flatten()

    # 获取相似度最高的k个文本的序号
    top_k_indices = similarities.argsort()[-k:][::-1]

    return top_k_indices.tolist()