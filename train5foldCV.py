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


def main(model, dataset, train_pairs, qrels, valid_run, test_run, model_out_dir, qrelDict, modelName, qidInWiki):
    LR = 0.001
    BERT_LR = 2e-5
    MAX_EPOCH = 1

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    top_valid_score = None
    bestResults = {}
    bestPredictions = []
    bestQids = []
    metricKeys = {"%s@%d" % (i, j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10, 15, 20]}
    metricKeys["rp"] = []

    for epoch in range(MAX_EPOCH):
        # loss = train_iteration(model, optimizer, dataset, train_pairs, qrels)
        # print(f'train epoch={epoch} loss={loss}')
        valid_qids, valid_results, valid_predictions = validate(model, dataset, valid_run, qrelDict, epoch, model_out_dir, qidInWiki)
        valid_score = np.mean(valid_results["ndcg@15"])
        print(f'validation epoch={epoch} score={valid_score}')
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            model.save(os.path.join(model_out_dir, 'weights.p'))
            test_qids, test_results, test_predictions = validate(model, dataset, test_run, qrelDict, epoch, model_out_dir, qidInWiki)
            bestResults = test_results
            bestPredictions = test_predictions
            bestQids = test_qids

#   save best results to files
    output = []
    for k in metricKeys:
        _res = np.mean(bestResults[k])
        print(_res, end="\t")
        output.append(str(_res))
    write2file("out5/", modelName, ".res", ",".join(output))


    # print(bestResults)
    # print(bestPredictions)
    # print()
    for k in metricKeys:
        result2file("out5/", modelName, "."+k, bestResults[k], bestQids)

    prediction2file("out5/", modelName, ".out", bestPredictions)

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


def validate(model, dataset, run, qrel, epoch, model_out_dir, qidInWiki):
    VALIDATION_METRIC = 'P.20'
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    return run_model(model, dataset, run, runf, qrel, qidInWiki)
    # return 0
    # return trec_eval(qrelf, runf)


def run_model(model, dataset, run, runf, qrels, qidInWiki, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
            # break

    res = {"%s@%d" %( i,j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10 ,15, 20]}
    res['rp'] = []
    predictions = []
    qids = []
    for qid in rerank_run:
        if int(qid) not in qidInWiki:
            continue
        ranked_list = [i[0] for i in sorted(rerank_run[qid].items(), key=lambda x: x[1], reverse=True)]
        for pid in ranked_list:
            predictions.append((qid, pid))
        result = eval(qrels[qid], ranked_list)
        for key in res:
            res[key].append(result[key])
        qids.append(qid)
    return qids, res, predictions



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
    thefile = open(path+name+format, 'w')
    for (qid, pid) in preds:
        thefile.write("%s\t%s\n" % (qid, pid))
    thefile.close()

def result2file(path, name, format, res, qids):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path+name+format, 'w')
    for q, r in zip(qids, res):
        thefile.write("%s\t%f\n" % (q, r))
    thefile.close()


def main_cli():

    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--data', default='query')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), default="data/cedr/query.tsv")
    parser.add_argument('--datafiles2', type=argparse.FileType('rt'), default="data/cedr/doc.tsv")
    parser.add_argument('--qrels', type=argparse.FileType('rt'), default="data/cedr/qrel.tsv")
    # parser.add_argument('--train_pairs', type=argparse.FileType('rt'), default="data/cedr/train0.tsv data/cedr/train1.tsv data/cedr/train2.tsv data/cedr/train3.tsv data/cedr/train4.tsv", nargs='+')
    # parser.add_argument('--valid_run', type=argparse.FileType('rt'), default="data/cedr/valid0.tsv data/cedr/valid1.tsv data/cedr/valid2.tsv data/cedr/valid3.tsv data/cedr/valid4.tsv", nargs='+')
    # parser.add_argument('--test_run', type=argparse.FileType('rt'), default="data/cedr/test0.tsv data/cedr/test1.tsv data/cedr/test2.tsv data/cedr/test3.tsv data/cedr/test4.tsv", nargs='+')
    parser.add_argument('--train_pairs', default="data/cedr/train")
    parser.add_argument('--valid_run', default="data/cedr/valid")
    parser.add_argument('--test_run', default="data/cedr/test")
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir', default="models/vbert")
    args = parser.parse_args()

    model = MODEL_MAP[args.model]().cuda() if data.device.type == 'cuda' else MODEL_MAP[args.model]()
    dataset = data.read_datafiles(args.datafiles, args.datafiles2)
    qrels = data.read_qrels_dict(args.qrels)

    train_pairs = []
    valid_run = []
    test_run = []
    for fold in range(5):
        f = open(args.train_pairs + "%d.tsv" % fold, "r")
        train_pairs.append(data.read_pairs_dict(f))
        f = open(args.valid_run + "%d.tsv" % fold, "r")
        valid_run.append(data.read_run_dict(f))
        f = open(args.test_run + "%d.tsv" % fold, "r")
        test_run.append(data.read_run_dict(f))

    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)

    timestamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    modelName = "%s_%s_%s_%s" % (args.model, args.data, "split", timestamp)

    df = pd.read_csv("data/cedr/qrel.tsv", sep="\t", names=["qid", "empty", "pid", "rele_label"])
    import collections
    qrelDict = collections.defaultdict(dict)
    for qid, prop, label in df[['qid', 'pid', 'rele_label']].values:
        qrelDict[str(qid)][str(prop)] = int(label)

    qidInWiki = pickle.load(open("qidInWiki", "rb"))

    for fold in range(len(train_pairs)):
        main(model, dataset, train_pairs[fold], qrels, valid_run[fold], test_run[fold], args.model_out_dir, qrelDict, modelName, qidInWiki)


if __name__ == '__main__':
    main_cli()
