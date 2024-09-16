import openai
import pandas as pd
import tqdm
from person_agent import PersonAgent
import json
import pickle
import os

domain = 'zhihu'
model_name = "gpt-4o-ca"
vector_path = 'data/twitter_user_history_m3e_matrix.pickle'
temperature = 0.1
max_topic = 10
batch_size = 4
history_window_size = 8
history_max_tokens = 40000
tag = f"recent_{history_window_size}_history"
openai_api_key = "sk-nZAG07Y65wMdwJRZWwlWJJcjCk8CcY7Eq2PDTSKFQRe84GVh"
openai_api_base = "https://api.chatanywhere.tech/v1"
#
# openai_api_key = "LL-dRr1Ys13cqf3tViSGCWUgM1JknmFm6HqnLYJJewiKQYS0u0v0OonvMyGaiYJuXFr"
# openai_api_base = "https://api.llama-api.com"

options_map = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)']

with open(vector_path, 'rb') as f:
    action_vectors = pickle.load(f)


def main():
    # Test Data Load
    qas = pd.read_csv(f'data/mini_{domain}_qa.csv')
    # qas = qas[qas['qtype'] == 'action_content']
    qas = qas.dropna(subset=['choices'])
    with open(f'data/{domain}_user_info.json', 'r') as f:
        users_info = json.load(f)
    save_path = f'predict_results/mini_{domain}_{model_name}_{tag}.xlsx'
    person_agent = PersonAgent(model_name=model_name, users_info=users_info, action_vectors=action_vectors,
                               apikey=openai_api_key, baseurl=openai_api_base,
                               temperature=temperature, max_topic=max_topic, history_window_size=history_window_size)

    if os.path.exists(save_path):
        checkpoint_save = pd.read_excel(save_path)
        all_res = checkpoint_save.to_dict(orient='records')
    else:
        all_res = []
    for i in tqdm.trange(len(all_res), qas.shape[0], batch_size):
        batch = qas[i:i+batch_size].to_dict('records')
        responses = person_agent.batch_predict([row['qtype'] for row in batch],
                                               [int(row['user_index']) for row in batch],
                                               [int(row['history_index']) for row in batch],
                                               [row['question'] for row in batch],
                                               [[f"{options_map[i]}.{option}" for i, option in enumerate(eval(row['choices']))] for row in batch],
                                               history_max_tokens)

        for response, row in zip(responses, batch):
            row['answer'] = options_map[int(row['answer_index'])]
            row['response'] = response
            row['predict_answer'] = response.get('predict_answer', None)
            all_res.append(row)
        if len(all_res) % (4 * batch_size) == 0:
            pd.DataFrame(all_res).to_excel(save_path, index=False)
    pd.DataFrame(all_res).to_excel(save_path, index=False)


if __name__ == '__main__':
    main()