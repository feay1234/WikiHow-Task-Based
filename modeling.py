import pickle

from pytools import memoize_method
import torch
import torch.nn.functional as F
import pytorch_pretrained_bert
from pytorch_pretrained_bert import BertModel

import modeling_util
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BertPairwiseRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1  # from bert-base-uncased
        self.BERT_SIZE = 768  # from bert-base-uncased
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask, prop_tok, prop_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 4  # = [CLS] and 3x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        CLSS = torch.full_like(query_tok[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_tok[:, :1], self.tokenizer.vocab['[SEP]'])
        TWOS = torch.zeros_like(query_mask[:, :1]) + 2
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_tok, SEPS, doc_tok, SEPS, prop_tok, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES, prop_mask, ONES], dim=1)
        segment_ids = torch.cat(
            [NILS] * (2 + QLEN) + [ONES] * (1 + doc_tok.shape[1]) + [TWOS] * (1 + prop_tok.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # TODO
        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class VanillaBertPairwiseRanker(BertPairwiseRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, prop_tok, prop_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask, prop_tok, prop_mask)
        return self.cls(self.dropout(cls_reps[-1]))


class BertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1  # from bert-base-uncased
        self.BERT_SIZE = 768  # from bert-base-uncased
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask, customBert=None):
        BATCH, QLEN = query_tok.shape
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, doc_toks, SEPS, query_toks, SEPS], dim=1)
        mask = torch.cat([ONES, doc_mask, ONES, query_mask, ONES], dim=1)
        # segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        segment_ids = torch.cat([NILS] * (2 + doc_toks.shape[1]) + [ONES] * (1 + QLEN), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # print(MAX_DOC_TOK_LEN, doc_tok.shape)

        # execute BERT model
        if not customBert:
            result = self.bert(toks, segment_ids.long(), mask)
        else:
            result = customBert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results
        # return cls_results, doc_results, query_results


class OriginalBertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1  # from bert-base-uncased
        self.BERT_SIZE = 768  # from bert-base-uncased
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class VanillaBertRanker(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        return self.cls(self.dropout(cls_reps[-1]))


class InvertBertRanker(BertRanker):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        # cls_reps, _, _ = self.encode_bert(doc_tok, doc_mask, query_tok, query_mask)
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        return self.cls(self.dropout(cls_reps[-1]))


class BirchRanker(BertRanker):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask, question_tok, question_mask):
        cls_query_tok, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        cls_doc_tok, _, _ = self.encode_bert(doc_tok, doc_mask, query_tok, query_mask)
        if self.args.mode % 2 == 0:
            cls_wiki_doc_tok, _, _ = self.encode_bert(wiki_tok, wiki_mask, doc_tok, doc_mask)
            cls_doc_wiki_tok, _, _ = self.encode_bert(doc_tok, doc_mask, wiki_tok, wiki_mask)

        if self.args.mode == 1:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            return self.cls(self.dropout(mul))

        elif self.args.mode == 2:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            mul_wiki = torch.mul(cls_wiki_doc_tok[-1], cls_doc_wiki_tok[-1])
            cat = self.cls(self.dropout(mul))
            cat_wiki = self.cls2(self.dropout(mul_wiki))
            return self.clsAll(torch.cat([cat, cat_wiki], dim=1))

class VanillaBirchtRanker(BirchRanker):
    def __init__(self, enableWiki, enableQuestion, shareBERT, args):
        super().__init__(enableWiki, enableQuestion, shareBERT)
        self.dropout = torch.nn.Dropout(0.1)
        self.args = args
        size = 0
        if enableWiki:
            size += 0
        if enableQuestion:
            size += 1
        # self.cls = torch.nn.Linear(self.BERT_SIZE * (1 + size), 1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        self.q = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)
        self.w = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask, question_tok, question_mask):
        cls_reps, _, _, cls_reps_wiki = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask,
                                                         question_tok,
                                                         question_mask)

        mul = torch.mul(self.q(cls_reps[-1]), self.w(cls_reps_wiki[-1]))

        # return self.cls(self.dropout(cls_reps[-1]))
        return self.cls(self.dropout(mul))


class CedrPacrrRanker(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()
        # QLEN = 20
        self.args = args
        QLEN = self.args.maxlen
        KMAX = 1  # Original was 2, which causes unknown bug
        NFILTERS = 32
        MINGRAM = 1
        MAXGRAM = 3
        self.simmat = modeling_util.SimmatModule()
        self.ngrams = torch.nn.ModuleList()
        self.rbf_bank = None
        for ng in range(MINGRAM, MAXGRAM + 1):
            ng = modeling_util.PACRRConvMax2dModule(ng, NFILTERS, k=KMAX, channels=self.CHANNELS)
            self.ngrams.append(ng)
        qvalue_size = len(self.ngrams) * KMAX
        self.linear1 = torch.nn.Linear(self.BERT_SIZE + QLEN * qvalue_size, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        scores = [ng(simmat) for ng in self.ngrams]
        scores = torch.cat(scores, dim=2)
        scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2])
        scores = torch.cat([scores, cls_reps[-1]], dim=1)
        rel = F.relu(self.linear1(scores))
        rel = F.relu(self.linear2(rel))
        rel = self.linear3(rel)
        return rel


class CedrKnrmRanker(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()
        self.args = args
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        # self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.kernels = modeling_util.KNRMRbfKernelBank(MUS, SIGMAS)
        self.combine = torch.nn.Linear(self.kernels.count() * self.CHANNELS + self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
            .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
            .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3)  # sum over document
        mask = (simmat.sum(dim=3) != 0.)  # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2)  # sum over query terms
        result = torch.cat([result, cls_reps[-1]], dim=1)
        scores = self.combine(result)  # linear combination over kernels
        return scores


class CedrDrmmRanker(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()
        self.args = args
        NBINS = 11
        HIDDEN = 5
        # self.bert_ranker = VanillaBertRanker(args)
        self.simmat = modeling_util.SimmatModule()
        self.histogram = modeling_util.DRMMLogCountHistogram(NBINS)
        self.hidden_1 = torch.nn.Linear(NBINS * self.CHANNELS + self.BERT_SIZE, HIDDEN)
        self.hidden_2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        histogram = self.histogram(simmat, doc_tok, query_tok)
        BATCH, CHANNELS, QLEN, BINS = histogram.shape
        histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
        # repeat cls representation for each query token
        cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
        output = torch.cat([output, cls_rep], dim=1)
        term_scores = self.hidden_2(torch.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        return term_scores.sum(dim=1)


class CustomBertModel(pytorch_pretrained_bert.BertModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but also outputs un-contextualized embeddings.
    """

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Based on pytorch_pretrained_bert.BertModel
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=True)

        return [embedding_output] + encoded_layers


class MSRanker(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()

        self.MS_SIZE = 100
        self.args = args

        self.text2MSvec = pickle.load(open("data/cedr/%s-ms%d" % (self.args.data.split("-")[0], self.args.maxlen), "rb"))

        self.dropout = torch.nn.Dropout(0.1)
        self.q = torch.nn.Linear(self.MS_SIZE, 100)
        self.d = torch.nn.Linear(self.MS_SIZE, 100)
        self.w = torch.nn.Linear(self.MS_SIZE, 100)
        self.qq = torch.nn.Linear(self.MS_SIZE, 100)
        self.cls = torch.nn.Linear(100, 1)

        self.properties = []

    def forward(self, query_tok, doc_tok, wiki_tok, question_tok):
        if self.args.mode == 1:
            mul = torch.mul(self.q(query_tok), self.d(doc_tok))
        elif self.args.mode == 2:
            mul = torch.mul(self.q(query_tok), self.d(doc_tok))
            mul = torch.mul(mul, self.w(wiki_tok))
        elif self.args.mode == 3:
            mul = torch.mul(self.q(query_tok), self.d(doc_tok))
            mul = torch.mul(mul, self.w(wiki_tok))
            # mul = torch.mul(mul, torch.max(self.qq(question_tok), 1).values)
            mul = torch.mul(mul, self.qq(question_tok))

        return self.cls(self.dropout(mul))

    @memoize_method
    def tokenize(self, text):
        text = " ".join(text.split(" ")[:self.args.maxlen]).replace("\'", "")
        if text in self.text2MSvec:
            return self.text2MSvec[text]
        print("not found:", text)
        return np.zeros(100)


class SentenceBert(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        self.cls2 = torch.nn.Linear(self.BERT_SIZE, 1)
        self.clsAll = torch.nn.Linear(2, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask, question_tok, question_mask):
        cls_query_tok, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        cls_doc_tok, _, _ = self.encode_bert(doc_tok, doc_mask, query_tok, query_mask)
        if self.args.mode in [2, 3, 4, 5, 6, 7, 9]:
            cls_wiki_doc_tok, _, _ = self.encode_bert(wiki_tok, wiki_mask, doc_tok, doc_mask)
            cls_doc_wiki_tok, _, _ = self.encode_bert(doc_tok, doc_mask, wiki_tok, wiki_mask)

        if self.args.mode == 1:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_doc_tok[-1]))
            return self.cls(self.dropout(mul))

        elif self.args.mode == 2:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_doc_tok[-1]))
            mul_wiki = torch.mul(self.dropout(cls_wiki_doc_tok[-1]), self.dropout(cls_doc_wiki_tok[-1]))
            cat = self.cls(self.dropout(mul))
            cat_wiki = self.cls2(self.dropout(mul_wiki))
            return self.clsAll(torch.cat([cat, cat_wiki], dim=1))

        elif self.args.mode == 3:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_doc_tok[-1]))
            mul_wiki = torch.mul(self.dropout(cls_wiki_doc_tok[-1]), self.dropout(cls_doc_wiki_tok[-1]))
            cat = self.cls(self.dropout(mul))
            cat_wiki = self.cls2(self.dropout(mul_wiki))
            return self.clsAll(self.dropout(torch.cat([cat, cat_wiki], dim=1)))

        elif self.args.mode == 4:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_doc_tok[-1]))
            mul_wiki = torch.mul(self.dropout(cls_wiki_doc_tok[-1]), self.dropout(cls_doc_wiki_tok[-1]))
            return self.cls(self.dropout(torch.mul(mul, mul_wiki)))

        elif self.args.mode == 5:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_wiki_doc_tok[-1]))
            mul_wiki = torch.mul(self.dropout(cls_doc_tok[-1]), self.dropout(cls_doc_wiki_tok[-1]))
            return self.cls(self.dropout(torch.mul(mul, mul_wiki)))

        elif self.args.mode == 6:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_doc_wiki_tok[-1]))
            mul_wiki = torch.mul(self.dropout(cls_doc_tok[-1]), self.dropout(cls_wiki_doc_tok[-1]))
            return self.cls(self.dropout(torch.mul(mul, mul_wiki)))

        elif self.args.mode == 7:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_doc_tok[-1]))
            mul_wiki = torch.mul(self.dropout(cls_wiki_doc_tok[-1]), self.dropout(cls_doc_wiki_tok[-1]))
            cat = self.cls(mul)
            cat_wiki = self.cls2(mul_wiki)
            return self.clsAll(self.dropout(torch.cat([cat, cat_wiki], dim=1)))

        if self.args.mode == 8:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            return self.cls(mul)

        elif self.args.mode == 9:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            mul_wiki = torch.mul(cls_wiki_doc_tok[-1], cls_doc_wiki_tok[-1])
            cat = self.cls(mul)
            cat_wiki = self.cls2(mul_wiki)
            return self.clsAll(torch.cat([cat, cat_wiki], dim=1))

class CrossBert(BertRanker):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        self.cls2 = torch.nn.Linear(self.BERT_SIZE, 1)
        self.clsAll = torch.nn.Linear(2, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask, question_tok, question_mask):
        cls_query_tok, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        cls_doc_tok, _, _ = self.encode_bert(doc_tok, doc_mask, query_tok, query_mask)
        if self.args.mode % 2 == 0:
            cls_wiki_doc_tok, _, _ = self.encode_bert(wiki_tok, wiki_mask, doc_tok, doc_mask)
            cls_doc_wiki_tok, _, _ = self.encode_bert(doc_tok, doc_mask, wiki_tok, wiki_mask)

        if self.args.mode == 1:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            return self.cls(self.dropout(mul))

        elif self.args.mode == 2:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            mul_wiki = torch.mul(cls_wiki_doc_tok[-1], cls_doc_wiki_tok[-1])
            cat = self.cls(self.dropout(mul))
            cat_wiki = self.cls2(self.dropout(mul_wiki))
            return self.clsAll(torch.cat([cat, cat_wiki], dim=1))

        elif self.args.mode == 3:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            return self.cls(mul)

        elif self.args.mode == 4:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            mul_wiki = torch.mul(cls_wiki_doc_tok[-1], cls_doc_wiki_tok[-1])
            cat = self.cls(mul)
            cat_wiki = self.cls2(mul_wiki)
            return self.clsAll(torch.cat([cat, cat_wiki], dim=1))

        elif self.args.mode == 5:
            mul = torch.mul(self.dropout(cls_query_tok[-1]), self.dropout(cls_doc_tok[-1]))
            return self.cls(self.dropout(mul))

        elif self.args.mode == 6:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            mul_wiki = torch.mul(cls_wiki_doc_tok[-1], cls_doc_wiki_tok[-1])
            cat = self.cls(mul)
            cat_wiki = self.cls2(mul_wiki)
            return self.clsAll(self.dropout(torch.cat([cat, cat_wiki], dim=1)))

class SBert(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        self.cls2 = torch.nn.Linear(self.BERT_SIZE, 1)
        self.clsAll = torch.nn.Linear(2, 1)

    def encode_sentence_bert(self, query_tok, query_mask):
        BATCH, QLEN = query_tok.shape

        query_toks = torch.cat([query_tok] , dim=0)
        query_mask = torch.cat([query_mask], dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS ], dim=1)
        mask = torch.cat([ONES, query_mask, ONES], dim=1)
        segment_ids = torch.cat([ONES] * (2 + QLEN), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # print(MAX_DOC_TOK_LEN, doc_tok.shape)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask, question_tok, question_mask):
        cls_query_tok = self.encode_sentence_bert(query_tok, query_mask)
        cls_doc_tok = self.encode_sentence_bert(doc_tok, doc_mask)
        if self.args.mode % 2 == 0:
            cls_wiki_doc_tok = self.encode_sentence_bert(wiki_tok, wiki_mask)
            cls_doc_wiki_tok = self.encode_sentence_bert(doc_tok, doc_mask)

        if self.args.mode == 1:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            return self.cls(self.dropout(mul))

        elif self.args.mode == 2:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            mul_wiki = torch.mul(cls_wiki_doc_tok[-1], cls_doc_wiki_tok[-1])
            cat = self.cls(self.dropout(mul))
            cat_wiki = self.cls2(self.dropout(mul_wiki))
            return self.clsAll(torch.cat([cat, cat_wiki], dim=1))

class MulBert(OriginalBertRanker):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        self.cls2 = torch.nn.Linear(self.BERT_SIZE, 1)
        self.w = torch.nn.Linear(1,1)
        self.clsAll = torch.nn.Linear(2, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask, question_tok, question_mask):
        cls_query_tok, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        cls_wiki_tok, _, _ = self.encode_bert(wiki_tok, wiki_mask, doc_tok, doc_mask)

        if self.args.mode == 1:
            mul = torch.mul(cls_query_tok[-1], cls_wiki_tok[-1])
            return self.cls(self.dropout(mul))
        elif self.args.mode == 3:
            return self.cls(self.dropout(cls_query_tok[-1])) + self.cls(self.dropout(cls_wiki_tok[-1]))
        elif self.args.mode == 4:
            return self.cls(cls_query_tok[-1]) + self.cls(cls_wiki_tok[-1])
        elif self.args.mode == 5:
            return self.cls(cls_query_tok[-1]) + self.cls2(cls_wiki_tok[-1])
        elif self.args.mode == 6:
            return self.cls(self.dropout(cls_query_tok[-1])) + self.w(self.cls(self.dropout(cls_wiki_tok[-1])))
        elif self.args.mode == 7:
            return self.cls(cls_query_tok[-1]) + self.w(self.cls(cls_wiki_tok[-1]))
        elif self.args.mode == 8:
            return self.cls(cls_query_tok[-1]) + self.w(self.cls2(cls_wiki_tok[-1]))
