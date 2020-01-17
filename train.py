import os
import argparse
import subprocess
import pandas as pd
import numpy as np
import random
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


def main(model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir, qrelDict):
    LR = 0.001
    BERT_LR = 2e-5
    MAX_EPOCH = 100

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    epoch = 0
    top_valid_score = None
    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss}')
        results = validate(model, dataset, valid_run, qrelDict, epoch, model_out_dir)
        # print(results)
        valid_score = np.mean(results["ndcg@15"])
        print(f'validation epoch={epoch} score={valid_score}')
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            model.save(os.path.join(model_out_dir, 'weights.p'))


def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 32
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
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


def validate(model, dataset, run, qrel, epoch, model_out_dir):
    VALIDATION_METRIC = 'P.20'
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    return run_model(model, dataset, run, runf, qrel)
    # return 0
    # return trec_eval(qrelf, runf)


def run_model(model, dataset, run, runf, qrels, desc='valid'):
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
    # with open(runf, 'wt') as runfile:
    #     for qid in rerank_run:
            # scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            # print(rerank_run[qid])
            # print(scores)
            # for i, (did, score) in enumerate(scores):
            #     runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

    res = {"%s@%d" %( i,j): [] for i in ["p", "r", "ndcg"] for j in [5, 10 ,15]}
    for qid in rerank_run:
        ranked_list = [i[0] for i in sorted(rerank_run[qid].items(), key=lambda x: x[1], reverse=True)]
        result = eval(qrels[qid], ranked_list)
        # print(result)
        # print()
        for key in res:
            res[key].append(result[key])
    return res



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

    for i in [5, 10, 15]:
        metric = MSnDCG(xrelnum, grades, cutoff=i)
        result["ndcg@%d" % i] = metric.compute(labeled_ranked_list)

        _ranked_list = ranked_list[:i]
        result["p@%d" % i] = len(set.intersection(set(qrels.keys()), set(_ranked_list))) / len(_ranked_list)
        result["r@%d" % i] = len(set.intersection(set(qrels.keys()), set(_ranked_list))) / len(qrels)

    return result

# def trec_eval(qrels, ranked_list):
#     trec_eval_f = '/Users/jarana/workspace/WikiHow-Task-Based/bin/trec_eval'
#     output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
#     output = output.replace('\t', ' ').split('\n')
#     assert len(output) == 1
#     return float(output[0].split()[2])



def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+', default="data/cedr/queries.tsv")
    parser.add_argument('--datafiles2', type=argparse.FileType('rt'), nargs='+', default="data/cedr/docs.tsv")
    parser.add_argument('--qrels', type=argparse.FileType('rt'), default="data/cedr/qrels")
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'), default="data/cedr/train.pairs")
    parser.add_argument('--valid_run', type=argparse.FileType('rt'), default="data/cedr/test.run")
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir', default="models/vbert")
    args = parser.parse_args()
    #model = MODEL_MAP[args.model]().cuda()
    model = MODEL_MAP[args.model]()
    dataset = data.read_datafiles(args.datafiles, args.datafiles2)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)

    df = pd.read_csv("data/cedr/qrels", sep="\t", names=["qid", "empty", "pid", "rele_label"])
    import collections
    qrelDict = collections.defaultdict(dict)
    for qid, prop, label in df[['qid', 'pid', 'rele_label']].values:
        qrelDict[str(qid)][str(prop)] = int(label)


    main(model, dataset, train_pairs, qrels, valid_run, args.qrels.name, args.model_out_dir, qrelDict)


if __name__ == '__main__':
    main_cli()
