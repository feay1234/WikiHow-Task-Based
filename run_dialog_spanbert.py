from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('bert', 'bert-large-uncased-whole-word-masking-finetuned-squad', use_cuda=torch.cuda.is_available()
, args={'reprocess_input_data': True, 'overwrite_output_dir': True})

# Train the model with JSON file
model.train_model('data/dialog/train.json')

# The list can also be used directly
# model.train_model(train_data)

# Evaluate the model. (Being lazy and evaluating on the train data itself)
result, text = model.eval_model('data/dialog/train.json')

print(result)
print(text)

print('-------------------')

# Making predictions using the model.
# to_predict = [{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What is this context?', 'id': '0'}]}]
#
# print(model.predict(to_predict))