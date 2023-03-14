from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('bert-base-nli-mean-tokens')

master_dict = [
                 'How to cancel my order?',
                 'Please let me know about the cancellation policy?',
                 'Do you provide refund?',
                 'what is the estimated delivery date of the product?',
                 'why my order is missing?',
                 'how do i report the delivery of the incorrect items?'
                 ]

inp_question = 'When is my product getting delivered?'

print('query : {}'.format(inp_question))

inp_question_representation = model.encode(inp_question, convert_to_tensor=True)

master_dict_representation = model.encode(master_dict, convert_to_tensor=True)

similarity = util.pytorch_cos_sim(inp_question_representation, master_dict_representation)

print('The most similar : {}'.format(master_dict[np.argmax(similarity)]))
