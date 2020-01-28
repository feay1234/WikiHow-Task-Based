import os
import argparse
import subprocess
from time import strftime, localtime

import pandas as pd
import numpy as np
import random ,pickle
from tqdm import tqdm
import torch
import modeling
import data
from pyNTCIREVAL import Labeler
from pyNTCIREVAL.metrics import MSnDCG, nERR, nDCG

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}


def main(model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir, qrelDict, modelName, qidInWiki, data):
    LR = 0.001
    BERT_LR = 2e-5
    MAX_EPOCH = 20

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    top_valid_score = None
    bestResults = {}
    metricKeys = {"%s@%d" % (i, j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10, 15, 20]}
    metricKeys["rp"] = []
    bestPredictions = []

    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss}')
        results, predictions = validate(model, dataset, valid_run, qrelDict, epoch, model_out_dir, qidInWiki, data)
        valid_score = np.mean(results["ndcg@15"])
        print(f'validation epoch={epoch} score={valid_score}')
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            # model.save(os.path.join(model_out_dir, 'weights.p'))
            print()
            output = []
            for k in metricKeys:
                _res = np.mean(results[k])
                print(_res, end="\t")
                output.append(str(_res))
            write2file("out/", modelName, ".out", ",".join(output))
            print()
            bestResults = results
            bestPredictions = predictions
#   save best results to file for t-test
    for k in metricKeys:
        result2file("out3/", modelName, "."+k, bestResults[k])

    prediction2file("out3/", modelName, ".out", bestPredictions)


def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 32
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            # print(record)
            # for i in record:
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, qrel, epoch, model_out_dir, qidInWiki, data):
    VALIDATION_METRIC = 'P.20'
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    return run_model(model, dataset, run, runf, qrel, qidInWiki, data)
    # return 0
    # return trec_eval(qrelf, runf)


def run_model(model, dataset, run, runf, qrels, qidInWiki, data, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE, data):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
            # break

    # with open(runf, 'wt') as runfile:
    #     for qid in rerank_run:
            # scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            # print(rerank_run[qid])
            # print(scores)
            # for i, (did, score) in enumerate(scores):
            #     runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

    res = {"%s@%d" %( i,j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10 ,15, 20]}
    res['rp'] = []
    predictions = []
    for qid in rerank_run:
        if int(qid) not in qidInWiki:
            continue
        ranked_list_scores = sorted(rerank_run[qid].items(), key=lambda x: x[1], reverse=True)
        ranked_list = [i[0] for i in ranked_list_scores]
        for (pid, score) in ranked_list_scores:
            predictions.append((qid, pid, score))
        result = eval(qrels[qid], ranked_list)
        for key in res:
            res[key].append(result[key])
    return res, predictions



def eval(qrels, ranked_list):
    # print(qrels)
    # print(ranked_list)
    # print()
    grades = [1, 2, 3, 4]  # a grade for relevance levels 1 and 2 (Note that level 0 is excluded)
    labeler = Labeler(qrels)
    labeled_ranked_list = labeler.label(ranked_list)
    rel_level_num = 5
    xrelnum = labeler.compute_per_level_doc_num(rel_level_num)
    result = {}

    for i in [5, 10, 15, 20]:
        metric = MSnDCG(xrelnum, grades, cutoff=i)
        result["ndcg@%d" % i] = metric.compute(labeled_ranked_list)

        nerr = nERR(xrelnum, grades, cutoff=i)
        result["nerr@%d" % i] = nerr.compute(labeled_ranked_list)

        _ranked_list = ranked_list[:i]
        result["p@%d" % i] = len(set.intersection(set(qrels.keys()), set(_ranked_list))) / len(_ranked_list)
        result["r@%d" % i] = len(set.intersection(set(qrels.keys()), set(_ranked_list))) / len(qrels)

    result["rp"] = len(set.intersection(set(qrels.keys()), set(ranked_list[:len(qrels)]))) / len(qrels)

    return result

def write2file(path, name, format, output):
    print(output)
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path+name+format, 'a')
    thefile.write("%s\n" % output)
    thefile.close()

def prediction2file(path, name, format, preds):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path+name+format, 'a')
    for (qid, pid, score) in preds:
        thefile.write("%s\t%s\t%f\n" % (qid, pid, score))
    thefile.close()

def result2file(path, name, format, res):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path+name+format, 'a')
    for r in res:
        thefile.write("%f\n" % r)
    thefile.close()




def main_cli():

    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--data', default='query')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), default="data/cedr/query.tsv")
    parser.add_argument('--datafiles2', type=argparse.FileType('rt'), default="data/cedr/doc.tsv")
    parser.add_argument('--qrels', type=argparse.FileType('rt'), default="data/cedr/qrel.tsv")
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'), default="data/cedr/train.tsv")
    parser.add_argument('--valid_run', type=argparse.FileType('rt'), default="data/cedr/test.tsv")
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir', default="models/vbert")
    args = parser.parse_args()

    model = MODEL_MAP[args.model]().cuda() if data.device.type == 'cuda' else MODEL_MAP[args.model]()
    dataset = data.read_datafiles(args.datafiles, args.datafiles2)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)

    # TODO support 5folds
    timestamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    modelName = "%s_%s_%s_%s" % (args.model, args.data, "split", timestamp)

    df = pd.read_csv("data/cedr/qrel.tsv", sep="\t", names=["qid", "empty", "pid", "rele_label"])
    import collections
    qrelDict = collections.defaultdict(dict)
    for qid, prop, label in df[['qid', 'pid', 'rele_label']].values:
        qrelDict[str(qid)][str(prop)] = int(label)

    qidInWiki = pickle.load(open("qidInWiki", "rb"))
    # print(qidInWiki)

    main(model, dataset, train_pairs, qrels, valid_run, args.qrels.name, args.model_out_dir, qrelDict, modelName, qidInWiki, data)
    # print(dataset)
    # maxlen = 0
    # for i in dataset[1]:
    #     l = len(dataset[1][i].split())
    #     maxlen = max(maxlen, l)
    # print(maxlen)


if __name__ == '__main__':
    main_cli()
