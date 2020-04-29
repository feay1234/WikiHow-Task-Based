import argparse
import json

from simpletransformers.question_answering import QuestionAnsweringModel
import logging
import torch, os

def print2file(path, name, format, printout,enablePrint=True):
    if enablePrint:
        print(printout)
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    thefile.write("%s\n" % (printout))
    thefile.close()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser('Dialog System with Transformers')
parser.add_argument('--model', type=str, default="bert")
parser.add_argument('--pre', type=str, default="squad")
parser.add_argument('--train', type=str, default="unseenTrain")
parser.add_argument('--test', type=str, default="unseenTest")
parser.add_argument('--out_dir', type=str, default="data/dialog/out/")
args = parser.parse_args()

pretrain = {'albert':'albert-xlarge-v2', 'bert':'bert-large-uncased', 'distilbert':'distilbert-base-uncased'}
pretrainSQUAD = {'albert':'ktrapeznikov/albert-xlarge-v2-squad-v2', 'bert':'bert-large-uncased-whole-word-masking-finetuned-squad', 'distilbert':'distilbert-base-uncased-distilled-squad'}
pretrainFile = pretrain[args.model] if args.pre == "" else pretrainSQUAD[args.model]

modelName = "%s_%s_%s_%s" % (args.model, pretrainFile.replace("/","-"), args.train, args.test)


# Create the QuestionAnsweringModel
model = QuestionAnsweringModel(args.model, pretrainFile, use_cuda=torch.cuda.is_available(), args={'reprocess_input_data': True, 'overwrite_output_dir': True})

# Train the model with JSON file
model.train_model('data/dialog/%s.json' % args.train)

# Evaluate the model. (Being lazy and evaluating on the train data itself)
result, out = model.eval_model('data/dialog/%s.json' % args.test)

print2file(args.out_dir, modelName, ".res", result, True)
# save output to file
with open('%s%s.out' % (args.out_dir, modelName), 'w') as f:
    json.dump(out, f)

print('-------------------')

# Making predictions using the model.
# to_predict = [{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What is this context?', 'id': '0'}]}]
#
# print(model.predict(to_predict))