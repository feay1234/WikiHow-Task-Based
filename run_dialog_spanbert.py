import argparse

from simpletransformers.question_answering import QuestionAnsweringModel
import logging
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser('Dialog System with Transformers')
parser.add_argument('--model', type=str)
parser.add_argument('--pre', type=str, default="")
args = parser.parse_args()

pretrain = {'albert':'albert-xlarge-v2', 'bert':'bert-large-uncased', 'distilbert':'distilbert-base-uncased'}
pretrainSQUAD = {'albert':'albert-xlarge-v2-squad-v2', 'bert':'bert-large-uncased-whole-word-masking-finetuned-squad', 'distilbert':'distilbert-base-uncased-distilled-squad'}
pretrainFile = pretrain[args.model] if args.pre == "" else pretrainSQUAD[args.model]


# Create the QuestionAnsweringModel
model = QuestionAnsweringModel(args.model, pretrainFile, use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})

# model = QuestionAnsweringModel('albert', 'albert-xlarge-v2', use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})
# model = QuestionAnsweringModel('albert', 'ktrapeznikov/albert-xlarge-v2-squad-v2', use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})
# model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased', use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})
# model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})
# model = QuestionAnsweringModel('bert', 'bert-large-uncased', use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})
# model = QuestionAnsweringModel('bert', 'bert-large-uncased-whole-word-masking-finetuned-squad', use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})


# # Train the model with JSON file
# model.train_model('data/dialog/train.json')
#
# # The list can also be used directly
# # model.train_model(train_data)
#
# # Evaluate the model. (Being lazy and evaluating on the train data itself)
# result, text = model.eval_model('data/dialog/test.json')
#
# print(result)
# print(text)
#
# print('-------------------')
#
# # Making predictions using the model.
# # to_predict = [{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What is this context?', 'id': '0'}]}]
# #
# # print(model.predict(to_predict))