import os
import argparse
import subprocess
from time import strftime, localtime

import pandas as pd
import numpy as np
import random, pickle
from tqdm import tqdm
import torch
import modeling
import Data
from pyNTCIREVAL import Labeler
from pyNTCIREVAL.metrics import MSnDCG, nERR, nDCG
import collections

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'bert': modeling.BERT,
    'sbert': modeling.SentenceBert,
    'ms': modeling.MSRanker,
    'birch': modeling.VanillaBirchtRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}


def main(model, dataset, train_pairs, qrels, valid_run, test_run, model_out_dir, qrelDict, modelName, qidInWiki, fold,
         metricKeys, MAX_EPOCH, data, args):
    LR = 0.001
    BERT_LR = 2e-5

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)
    # optimizer = torch.optim.Adam([non_bert_params], lr=LR)

    top_valid_score = None
    bestResults = {}
    bestPredictions = []
    bestQids = []

    print("Fold: %d" % fold)

    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels, data, args)
        print(f'train epoch={epoch} loss={loss}')
        valid_qids, valid_results, valid_predictions = validate(model, dataset, valid_run, qrelDict, epoch,
                                                                model_out_dir, qidInWiki, data, args, "valid")
        valid_score = np.mean(valid_results["ndcg@15"])
        print(f'validation epoch={epoch} score={valid_score}')
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score')
            # model.save(os.path.join(model_out_dir, 'weights.p'))
            test_qids, test_results, test_predictions = validate(model, dataset, test_run, qrelDict, epoch,
                                                                 model_out_dir, qidInWiki, data, args, "test")
            bestResults = test_results
            bestPredictions = test_predictions
            bestQids = test_qids

    #   save outputs to files

    for k in metricKeys:
        result2file(args.out_dir, modelName, "." + k, bestResults[k], bestQids, fold)

    prediction2file(args.out_dir, modelName, ".out", bestPredictions, fold)
    return bestResults


def train_iteration(model, optimizer, dataset, train_pairs, qrels, data, args):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 32
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in Data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE, data, args):

            if isinstance(model, modeling.BirchRanker):
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'],
                               record['wiki_tok'],
                               record['wiki_mask'],
                               record['question_tok'],
                               record['question_mask'])
            elif args.model in ["ms", "sbert"]:
                scores = model(record['query_tok'],
                               record['doc_tok'],
                               record['wiki_tok'],
                               record['question_tok'])
            else:
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'])

            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])  # pariwse softmax
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, qrel, epoch, model_out_dir, qidInWiki, data, args, desc):
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    return run_model(model, dataset, run, runf, qrel, qidInWiki, data, args, desc)


def run_model(model, dataset, run, runf, qrels, qidInWiki, data, args, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in Data.iter_valid_records(model, dataset, run, BATCH_SIZE, data, args):
            if isinstance(model, modeling.BirchRanker):
                scores = model(records['query_tok'],
                               records['query_mask'],
                               records['doc_tok'],
                               records['doc_mask'],
                               records['wiki_tok'],
                               records['wiki_mask'],
                               records['question_tok'],
                               records['question_mask'])
            elif args.model in ["ms", "sbert"]:
                scores = model(records['query_tok'],
                               records['doc_tok'],
                               records['wiki_tok'],
                               records['question_tok'])
            else:
                scores = model(records['query_tok'],
                               records['query_mask'],
                               records['doc_tok'],
                               records['doc_mask'])

            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
            # break

    res = {"%s@%d" % (i, j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10, 15, 20]}
    res['rp'] = []
    predictions = []
    qids = []
    for qid in rerank_run:
        if args.evalMode != "all" and int(qid) not in qidInWiki:
            continue
        ranked_list_scores = sorted(rerank_run[qid].items(), key=lambda x: x[1], reverse=True)
        ranked_list = [i[0] for i in ranked_list_scores]
        for (pid, score) in ranked_list_scores:
            predictions.append((qid, pid, score))
        result = eval(qrels[qid], ranked_list)
        for key in res:
            res[key].append(result[key])
        qids.append(qid)
    return qids, res, predictions


def eval(qrels, ranked_list):
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
    thefile = open(path + name + format, 'a')
    thefile.write("%s\n" % output)
    thefile.close()


def prediction2file(path, name, format, preds, fold):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    for (qid, pid, score) in preds:
        thefile.write("%d\t%s\t%s\t%f\n" % (fold, qid, pid, score))
    thefile.close()


def result2file(path, name, format, res, qids, fold):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    for q, r in zip(qids, res):
        thefile.write("%d\t%s\t%f\n" % (fold, q, r))
    thefile.close()

    # 'cedr_pacrr': modeling.CedrPacrrRanker,
    # 'cedr_knrm': modeling.CedrKnrmRanker,
    # 'cedr_drmm': modeling.CedrDrmmRanker


def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='sbert')
    parser.add_argument('--data', default='query')
    # parser.add_argument('--datafiles', type=argparse.FileType('rt'), default="data/cedr/query-title-bm25-v2.tsv")
    parser.add_argument('--queryfile', type=argparse.FileType('rt'), default="data/cedr/query.tsv")
    parser.add_argument('--docfile', type=argparse.FileType('rt'), default="data/cedr/doc.tsv")
    parser.add_argument('--wikifile', type=argparse.FileType('rt'), default="data/cedr/wikipedia1.tsv")
    parser.add_argument('--questionfile', type=argparse.FileType('rt'), default="data/cedr/question-qq5.tsv")

    parser.add_argument('--qrels', type=argparse.FileType('rt'), default="data/cedr/qrel.tsv")
    parser.add_argument('--train_pairs', default="data/cedr/train")
    parser.add_argument('--valid_run', default="data/cedr/valid")
    parser.add_argument('--test_run', default="data/cedr/test")
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir', default="models/vbert")
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--out_dir', default="out/")
    parser.add_argument('--evalMode', default="all")
    parser.add_argument('--mode', type=int, default=1)

    args = parser.parse_args()

    if args.model == "birch":
        if args.mode == 1:
            model = MODEL_MAP[args.model](True, False, False, args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[
                args.model](True, False, False, args)
        elif args.mode == 2:
            model = MODEL_MAP[args.model](False, True, False, args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[
                args.model](False, True, False, args)
        elif args.mode == 3:
            model = MODEL_MAP[args.model](True, False, True, args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[
                args.model](True, False, True, args)
        elif args.mode == 4:
            model = MODEL_MAP[args.model](False, True, True, args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[
                args.model](False, True, True, args)
        elif args.mode == 5:
            model = MODEL_MAP[args.model](True, True, True, args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[
                args.model](True, True, True, args)
        elif args.mode == 6:
            model = MODEL_MAP[args.model](True, True, False, args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[
                args.model](True, True, False, args)
    elif args.model in ["ms", "sbert"]:
        model = MODEL_MAP[args.model](args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[args.model](args)
    else:
        model = MODEL_MAP[args.model]().cuda() if Data.device.type == 'cuda' else MODEL_MAP[args.model]()
    dataset = Data.read_datafiles([args.queryfile, args.docfile, args.wikifile,
                                   args.questionfile] if args.model in ["birch", "ms", "sbert"] else [
        args.queryfile, args.docfile])

    if isinstance(model, modeling.CedrPacrrRanker):
        args.maxlen = min(500, max([len(model.tokenize(dataset[0][i])) for i in dataset[0]]))
        model = MODEL_MAP[args.model](args.maxlen).cuda() if Data.device.type == 'cuda' else MODEL_MAP[args.model](
            args.maxlen)

    qrels = Data.read_qrels_dict(args.qrels)

    MAX_EPOCH = args.epoch

    train_pairs = []
    valid_run = []
    test_run = []

    foldNum = args.fold
    for fold in range(foldNum):
        f = open(args.train_pairs + "%d.tsv" % fold, "r")
        train_pairs.append(Data.read_pairs_dict(f))
        f = open(args.valid_run + "%d.tsv" % fold, "r")
        valid_run.append(Data.read_run_dict(f))
        f = open(args.test_run + "%d.tsv" % fold, "r")
        test_run.append(Data.read_run_dict(f))

    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    timestamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    if "birch" in args.model:
        wikiName = args.wikifile.name.split("/")[-1].replace(".tsv", "")
        questionName = args.questionfile.name.split("/")[-1].replace(".tsv", "")
        additionName = []
        if args.mode in [1, 3, 5, 6]:
            additionName.append(wikiName)
        if args.mode in [2, 4, 5, 6]:
            additionName.append(questionName)

        modelName = "%s_m%d_%s_%s_%s_e%d_%s" % (
            args.model, args.mode, args.data, "_".join(additionName), args.evalMode, args.epoch, timestamp)
    elif args.model in ["ms", "sbert"]:
        modelName = "%s_m%d_%s_%s_e%d_%s" % (args.model, args.mode, args.data, args.evalMode, args.epoch, timestamp)
    else:
        modelName = "%s_%s_%s_e%d_%s" % (args.model, args.data, args.evalMode, args.epoch, timestamp)
    print(modelName)

    df = pd.read_csv("data/cedr/qrel.tsv", sep="\t", names=["qid", "empty", "pid", "rele_label"])
    qrelDict = collections.defaultdict(dict)
    for qid, prop, label in df[['qid', 'pid', 'rele_label']].values:
        qrelDict[str(qid)][str(prop)] = int(label)

    qidInWiki = pickle.load(open("qidInWiki", "rb"))

    metricKeys = {"%s@%d" % (i, j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10, 15, 20]}
    metricKeys["rp"] = []

    results = []
    for fold in range(len(train_pairs)):
        results.append(
            main(model, dataset, train_pairs[fold], qrels, valid_run[fold], test_run[fold], args.model_out_dir,
                 qrelDict, modelName, qidInWiki, fold, metricKeys, MAX_EPOCH, Data, args))

    #   average results across 5 folds
    output = []
    for k in metricKeys:
        tmp = []
        for fold in range(foldNum):
            tmp.extend(results[fold][k])
        _res = np.mean(tmp)
        output.append("%.4f" % _res)
    write2file(args.out_dir, modelName, ".res", ",".join(output))


if __name__ == '__main__':
    main_cli()
