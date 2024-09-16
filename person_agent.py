from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import re
# from vector_model import find_top_k_similar, get_m3e_vectors

def get_key_or_string(input_dict, key):
    if isinstance(input_dict, dict) and key in input_dict:
        return input_dict[key]
    else:
        return input_dict

class PersonAgent:
    def __init__(self, model_name, users_info, action_vectors, temperature=0.1, max_topic=10, history_window_size=8,
                 baseurl='https://api.chatanywhere.tech/v1', apikey='sk-nZAG07Y65wMdwJRZWwlWJJcjCk8CcY7Eq2PDTSKFQRe84GVh'):
        self.model = ChatOpenAI(model=model_name,
                                openai_api_key=apikey,
                                openai_api_base=baseurl,
                                temperature=temperature)
        self.parser = StrOutputParser()
        self.max_topic = max_topic
        self.users_info = users_info
        self.history_window_size = history_window_size
        self.action_vectors = action_vectors
        # self.example = {"thinking": "the thinking process for answer",
        #                 "predict_answer_option": "final predicted answer option based on previous thinking process"}

    @property
    def twitter_prompt(self):
        template = """
        You are an expert at analyzing user behavior and making inferences.
        Given a user's personal information and behavior history sequence (it may also be empty), along with a question and options, please predict the most likely answer.
        
        ### User Information
        Username: {user_name}
        Interested Topics: {topics}
        User Description: {description}
        User Behavior History:
        {history}
        
        ### Question & Options
        Question: {question}
        Options: {choices}
        
        ### Instruction
        Please select the only one most possible answer from above options about the question.
        You need to conduct a thorough analysis of the options, examining the user's profile and historical interaction records to infer their preferences and psychology. Based on this analysis, make informed assumptions about the user's most likely interaction behaviors.
        
        output example: 
        ### Thinking
        the thinking process for answer
        ### Predict Answer
        Therefore the answer is (your choose option).
        
        Output: 
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["user_name", "topics", "description", "history", "question", "choices"],
            pattern=re.compile(r"\`\`\`\n\`\`\`")
        )
        return prompt

    @property
    def zhihu_prompt(self):
        template = """
            用户信息
            用户名: {user_name}
            感兴趣的主题: {topics}
            用户描述: {description}
            用户行为历史:
            {history}
            
            问题与选项
            问题: {question}
            选项: {choices}
            
            ### Instruction
            请从上述选项中选择一个最有可能的答案来回答问题。你需要对选项进行深入分析，结合用户的个人资料和历史互动记录，推断他们的偏好和心理。基于这种分析，做出关于用户最可能的互动行为的合理假设。
            
            output example:
            ### Thinking
            得到答案的思考过程
            ### Predict Answer
            Therefore the answer is (your choose option).
            output:
            """
        prompt = PromptTemplate(
            template=template,
            input_variables=["user_name", "topics", "description", "history", "question", "choices"],
            pattern=re.compile(r"\`\`\`\n\`\`\`")
        )
        return prompt

    # def parse(self, response):
    #     original_response = response
    #     try:
    #         response = eval(response)
    #     except Exception as e:
    #         print(e)
    #         response = {
    #             'thinking': "none",
    #             'predict_answer_option': "none",
    #         }
    #     res = {"response": original_response}
    #     res.update(response)
    #     return res

    def parse(self, response):
        pattern = r"Therefore.*?the answer is \((.*?)\)"
        match = re.search(pattern, response)
        if match is None:
            answer = None
        else:
            answer = match.group(1)
        res = {"response": response, "predict_answer": '({})'.format(answer)}
        return res

    def recent_window_history(self, all_history, history_index):
        history = all_history[:history_index]
        return history[-self.history_window_size:]

    # def similar_history(self, all_history, history_index, user_index, choices, qtype):
    #     history = all_history[:history_index]
    #     if qtype == 'action_type' or history_index <= self.history_window_size:
    #         return history[-self.history_window_size:]
    #
    #     k = self.history_window_size // 4
    #     chain_history = []
    #     for choice in choices:
    #         choice_vector = get_m3e_vectors(choice)
    #         history_matrix = self.action_vectors[user_index][:history_index, :]
    #         similar_index = find_top_k_similar(history_matrix, choice_vector, k)
    #         similar_history = [history[i] for i in similar_index]
    #         chain_history.extend(similar_history)
    #
    #     return chain_history

    def build_chain_args(self, qtype, user_index, history_index, question, choices, history_max_tokens):
        user_info = self.users_info[user_index]['user_info']
        history = self.recent_window_history(self.users_info[user_index]['actions'], history_index)
        # history = self.similar_history(self.users_info[user_index]['actions'], history_index, user_index, choices, qtype)

        for h in history:
            h['action_object'] = h['action_object'][:history_max_tokens] if h['action_object'] is not None else None
            h['action_content'] = h['action_content'][:history_max_tokens] if h['action_content'] is not None else None

        chain_args = {"user_name": user_info['username'],
                      "topics": user_info['topics'][:self.max_topic],
                      "description": user_info['description'],
                      "history": history,
                      "choices": choices,
                      "question": question}
        return chain_args

    def predict(self,qtype, user_index, history_index, question, choices, history_max_tokens):
        chain = self.zhihu_prompt | self.model | self.parser
        try:
            chain_args = self.build_chain_args(qtype, user_index, history_index, question, choices, history_max_tokens)
            response = chain.invoke(chain_args)
            response = self.parse(response)
        except Exception as e:
            print(e)
            chain_args = self.build_chain_args(qtype, user_index, history_index, question, choices, history_max_tokens // 2)
            response = chain.invoke(chain_args)
            response = self.parse(response)
        return response

    def batch_predict(self,batch_qtype, batch_user_index, batch_history_index, batch_question, batch_choices, batch_size=8, history_max_tokens=400):
        chain = self.zhihu_prompt | self.model | self.parser
        all_chain_args = []
        for qtype, user_index, history_index, question, choices in zip(batch_qtype, batch_user_index, batch_history_index, batch_question, batch_choices):
            chain_args = self.build_chain_args(qtype, user_index, history_index, question, choices, history_max_tokens)
            all_chain_args.append(chain_args)
        try:
            responses = chain.batch(all_chain_args, max_concurrency=batch_size)
            responses = [self.parse(response) for response in responses]
        except Exception as e:
            print(e)
            responses = []
            for chain_args in all_chain_args:
                try:
                    response = chain.invoke(chain_args)
                    response = self.parse(response)
                    responses.append(response)
                except Exception as e:
                    response = {"response": None, "predict_answer": None}
                    print(e)
                responses.append(response)
        return responses