import random
from tqdm import tqdm
import torch
import numpy as np
import modeling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_datafiles(files):
    queries, wikis, questions, docs, qtypes = {}, {}, {}, {}, {}
    # for file in files:
    for idx, file in enumerate(files):
        for line in tqdm(file, desc='loading datafile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) == 3:
                c_type, c_id, c_text = cols
            elif len(cols) == 4:
                c_type, c_id, c_text, c_qtype = cols
            # assert c_type in ('query', 'doc', 'wiki', 'question')
            # if c_type == 'query':
            if idx == 0:
                queries[c_id] = c_text
                qtypes[c_id] = c_qtype
            # elif c_type == 'doc':
            elif idx == 1:
                docs[c_id] = c_text
            elif idx == 2:
                wikis[c_id] = c_text
            elif idx == 3:
                questions[c_id] = c_text

    return queries, docs, wikis, questions, qtypes


def read_qrels_dict(file):
    result = {}
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score, _ = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def read_run_dict(file):
    result = {}
    for line in tqdm(file, desc='loading run (by line)', leave=False):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = float(score)
    return result


def read_pairs_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid, docid = line.split()
        result.setdefault(qid, {})[docid] = 1
    return result


def iter_train_pairs(model, dataset, train_pairs, qrels, batch_size, data, args):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'wiki_tok': [], 'question_tok': [], 'label': [], 'restriction': []}
    if args.model == "sigir_sota":
        for qid, query_tok, etype, label in _iter_train_pairs(model, dataset, train_pairs, qrels,
                                                                                      args):
            batch['query_id'].append(qid)
            batch['query_tok'].append(query_tok)
            tmp = np.zeros(model.propNum)
            tmp[list(args.type2pids[etype])] = 1
            batch['restriction'].append(tmp)
            tmp = np.zeros(model.propNum)
            tmp[[int(i) for i in label]] = 1
            batch['label'].append(tmp)

            if len(batch['query_id']) // 2 == batch_size:
                yield _pack_n_ship(batch, data, args)
                batch = {'query_id': [], 'query_tok': [], 'label': [], 'restriction': []}
    else:
        for qid, did, query_tok, doc_tok, wiki_tok, question_tok in _iter_train_pairs(model, dataset, train_pairs, qrels,
                                                                                      args):
            batch['query_id'].append(qid)
            batch['doc_id'].append(did)
            batch['query_tok'].append(query_tok)
            batch['doc_tok'].append(doc_tok)
            batch['wiki_tok'].append(wiki_tok)
            batch['question_tok'].append(question_tok)

            if len(batch['query_id']) // 2 == batch_size:
                yield _pack_n_ship(batch, data, args)
                batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'wiki_tok': [], 'question_tok': []}


def _iter_train_pairs(model, dataset, train_pairs, qrels, args):
    ds_queries, ds_docs, ds_wikis, ds_questions, ds_qtypes = dataset
    while True:
        qids = list(train_pairs.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                continue

            if args.model == "sigir_sota":
                yield qid, model.tokenize(ds_queries[qid]), ds_qtypes[qid], pos_ids
            else:
                pos_id = random.choice(pos_ids)
                neg_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) == 0]

                if len(neg_ids) == 0:
                    print("No neg instances", qid)
                    continue
                neg_id = random.choice(neg_ids)
                query_tok = model.tokenize(ds_queries[qid])
                wiki_tok = model.tokenize(ds_wikis[qid])
                question_tok = model.tokenize(ds_questions[qid])

                pos_doc = ds_docs.get(pos_id)
                neg_doc = ds_docs.get(neg_id)
                if pos_doc is None:
                    tqdm.write(f'missing doc {pos_id}! Skipping')
                    continue
                if neg_doc is None:
                    tqdm.write(f'missing doc {neg_id}! Skipping')
                    continue

                yield qid, pos_id, query_tok, model.tokenize(pos_doc), wiki_tok, question_tok
                yield qid, neg_id, query_tok, model.tokenize(neg_doc), wiki_tok, question_tok

        # break


def iter_valid_records(model, dataset, run, batch_size, data, args):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'wiki_tok': [], 'question_tok': [], 'label': [], 'restriction':[] }

    for qid, did, query_tok, doc_tok, wiki_tok, question_tok in _iter_valid_records(model, dataset, run, args):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        batch['wiki_tok'].append(wiki_tok)
        batch['question_tok'].append(question_tok)

        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(batch, data, args)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'wiki_tok': [], 'question_tok': []}

    # final batch
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(batch, data, args)


def _iter_valid_records(model, dataset, run, args):
    ds_queries, ds_docs, ds_wikis, ds_questions, ds_qtypes = dataset
    for qid in run:
        query_tok = model.tokenize(ds_queries[qid])
        wiki_tok = model.tokenize(ds_wikis[qid])
        question_tok = model.tokenize(ds_questions[qid])

        if args.model == "sigir_sota":
            yield qid, model.tokenize(ds_queries[qid]), ds_qtypes[qid], run[qid]
        else:
            for did in run[qid]:
                doc = ds_docs.get(did)
                if doc is None:
                    tqdm.write(f'missing doc {did}! Skipping')
                    continue
                doc_tok = model.tokenize(doc)
                yield qid, did, query_tok, doc_tok, wiki_tok, question_tok


def _pack_n_ship(batch, data, args):
    # QLEN = 9
    # MAX_DLEN = 800
    # DLEN = min(MAX_DLEN, max(len(b) for b in batch['query_tok']))

    if "birch" in args.model:
        DLEN = min(512, int(np.max([len(b) for b in batch['query_tok']])))
        WLEN = min(512, int(np.max([len(b) for b in batch['wiki_tok']])))
        QQLEN = min(512, int(np.max([len(b) for b in batch['question_tok']])))
        QLEN = 9 if not args.model == "cedr_pacrr" else args.maxlen

        return {
            'query_id': batch['query_id'],
            'doc_id': batch['doc_id'],
            'query_tok': _pad_crop(batch['doc_tok'], QLEN),
            'doc_tok': _pad_crop(batch['query_tok'], DLEN),
            'wiki_tok': _pad_crop(batch['wiki_tok'], WLEN),
            'question_tok': _pad_crop(batch['question_tok'], QQLEN),
            'query_mask': _mask(batch['doc_tok'], QLEN),
            'doc_mask': _mask(batch['query_tok'], DLEN),
            'wiki_mask': _mask(batch['wiki_tok'], WLEN),
            'question_mask': _mask(batch['question_tok'], QQLEN),
        }

    elif args.model in ["ms", "sent_transformer", "unsup"]:

        return {
            'query_id': batch['query_id'],
            'doc_id': batch['doc_id'],
            'query_tok': toTensor(batch['query_tok']),
            'doc_tok': toTensor(batch['doc_tok']),
            'wiki_tok': toTensor(batch['wiki_tok']),
            # 'question_tok': _pad_crop_np(batch['question_tok'], 5),
            'question_tok': toTensor(batch['question_tok'])
        }
    elif args.model in ["sbert", "crossbert", "crossbert2", "mulbert", "mul_cedr_drmm", "mul_cedr_knrm", "mul_cedr_pacrr"]:

        QLEN = min(args.maxlen, int(np.max([len(b) for b in batch['query_tok']])))
        DLEN = min(args.maxlen, int(np.max([len(b) for b in batch['doc_tok']])))
        WLEN = min(args.maxlen, int(np.max([len(b) for b in batch['wiki_tok']])))
        QQLEN = min(args.maxlen, int(np.max([len(b) for b in batch['question_tok']])))

        if args.model == "mul_cedr_pacrr":
            QLEN = args.maxlen
            DLEN = args.maxlen
            WLEN = args.maxlen
        # QLEN = args.maxlen
        # DLEN = args.maxlen
        # WLEN = args.maxlen
        # QQLEN = args.maxlen

        return {
            'query_id': batch['query_id'],
            'doc_id': batch['doc_id'],
            'query_tok': _pad_crop(batch['query_tok'], QLEN),
            'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
            'wiki_tok': _pad_crop(batch['wiki_tok'], WLEN),
            'question_tok': _pad_crop(batch['question_tok'], QQLEN),
            'query_mask': _mask(batch['query_tok'], QLEN),
            'doc_mask': _mask(batch['doc_tok'], DLEN),
            'wiki_mask': _mask(batch['wiki_tok'], WLEN),
            'question_mask': _mask(batch['question_tok'], QQLEN),
        }
    elif args.model == "sigir_sota":
        QLEN = min(args.maxlen, int(np.max([len(b) for b in batch['query_tok']])))
        return {
            'query_id': batch['query_id'],
            'query_tok': _pad_crop(batch['query_tok'], QLEN),
            'query_mask': _mask(batch['query_tok'], QLEN),
            'label': batch['label'],
            'restriction': batch['restriction']
        }
    else:

        if args.mode == 1:
            QLEN = args.maxlen
            MAX_DLEN = 800
            DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
            # QLEN = min(args.maxlen,
            #            int(np.max([len(b) for b in batch['query_tok']]))) if args.model != "cedr_pacrr" else 20
            # DLEN = min(args.maxlen, int(np.max([len(b) for b in batch['doc_tok']])))
            return {
                'query_id': batch['query_id'],
                'doc_id': batch['doc_id'],
                'query_tok': _pad_crop(batch['query_tok'], QLEN),
                'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
                'query_mask': _mask(batch['query_tok'], QLEN),
                'doc_mask': _mask(batch['doc_tok'], DLEN),
            }
        elif args.mode in [2, 3]:
            QLEN = min(args.maxlen,
                       int(np.max([len(b) for b in batch['query_tok']]))) if args.model != "cedr_pacrr" else args.maxlen
            DLEN = min(args.maxlen, int(np.max([len(b) for b in batch['doc_tok']])))
            WLEN = min(args.maxlen, int(np.max([len(b) for b in batch['wiki_tok']])))
            QQLEN = min(args.maxlen, int(np.max([len(b) for b in batch['question_tok']])))

            toks = []
            if args.mode == 2:
                for i, j in zip(batch['query_tok'], batch['wiki_tok']):
                    toks.append(i + j)
                QLEN = QLEN + WLEN if args.model != "cedr_pacrr" else args.maxlen
                # print(len(batch['query_tok'][0]), len(toks[0]))
            elif args.mode == 3:
                for i, j, k in zip(batch['query_tok'], batch['wiki_tok'], batch['question_tok']):
                    toks.append(i + j + k)
                QLEN = QLEN + WLEN + QQLEN if args.model != "cedr_pacrr" else args.maxlen
            return {
                'query_id': batch['query_id'],
                'doc_id': batch['doc_id'],
                'query_tok': _pad_crop(toks, QLEN),
                'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
                'query_mask': _mask(toks, QLEN),
                'doc_mask': _mask(batch['doc_tok'], DLEN),
            }


def toTensor(x):
    # print(torch.tensor(x))
    # try:
    return torch.tensor(x).float().cuda() if device.type == 'cuda' else torch.tensor(x).float()
    # except:
    #     print(x)


def _pad_crop_np(items, l):
    results = []
    for item in items:
        if len(item) < l:
            while len(item) != l:
                item.append([0] * 100)
        if len(item) > l:
            item = item[:l]
        results.append(item)
    return torch.tensor(results).float().cuda() if device.type == 'cuda' else torch.tensor(results).float()


def _pad_crop(items, l, val=-1):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [val] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    return torch.tensor(result).long().cuda() if device.type == 'cuda' else torch.tensor(result).long()


def _mask(items, l):
    result = []
    for item in items:
        # needs padding (masked)
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        # no padding (possible crop)
        if len(item) >= l:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    return torch.tensor(result).float().cuda() if device.type == 'cuda' else torch.tensor(result).float()
