import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel
import os

def count_frequency(triples, start=4):
    count = {}
    for head, relation, tail in triples:
        if (head, relation) not in count:
            count[(head, relation)] = start
        else:
            count[(head, relation)] += 1

        if (tail, -relation-1) not in count:
            count[(tail, -relation-1)] = start
        else:
            count[(tail, -relation-1)] += 1
    return count


def concept_filter_h(head, relation, rel_h, rel2nn, ent_conc):
    if str(relation) not in rel_h:
        return []
    rel_hc = rel_h[str(relation)]
    set_hc = set(rel_hc)
    h = []
    if rel2nn[str(relation)] == 0 or rel2nn[str(relation)] == 1:
        if str(head) not in ent_conc:
            for hc in rel_hc:
                for ent in conc_ents[str(hc)]:
                    h.append(ent)
        else:
            for conc in ent_conc[str(head)]:
                for ent in conc_ents[str(conc)]:
                    h.append(ent)
    else:
        if str(head) in ent_conc:
            set_ent_conc = set(ent_conc[str(head)])
        else:
            set_ent_conc = set([])
        set_diff = set_hc - set_ent_conc
        set_diff = list(set_diff)
        for conc in set_diff:
            for ent in conc_ents[str(conc)]:
                h.append(ent)

    h = set(h)
    return list(h)


def concept_filter_t(tail, relation, rel_t, rel2nn, ent_conc):
    '''
    filter the concepts by commonsense
    '''
    if str(relation) not in rel_t:
        return []
    rel_tc = rel_t[str(relation)]
    set_tc = set(rel_tc)
    t = []
    if rel2nn[str(relation)] == 2 or rel2nn[str(relation)] == 3:
        if tail in ent_conc:
            for conc in ent_conc[str(tail)]:
                for ent in conc_ents[str(conc)]:
                    t.append(ent)
        else:
            for tc in rel_tc:
                for ent in conc_ents[str(tc)]:
                    t.append(ent)
    else:
        if str(tail) in ent_conc:
            set_ent_conc = set(ent_conc[str(tail)])
        else:
            set_ent_conc = set([])
        set_diff = set_tc - set_ent_conc
        set_diff = list(set_diff)
        for conc in set_diff:
            for ent in conc_ents[str(conc)]:
                t.append(ent)
    t = set(t)
    return list(t)


def get_true_head_and_tail(triples):
    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

    return true_head, true_tail


def _process(triples, pid,rel_h,nentity,ent_conc,all_triples):
    res = []
    for trip in tqdm(triples, desc="PID:" + str(pid)):
        head, relation, tail = trip
        set_rel2h = set(rel_h[str(relation)])

        tmp = []
        for rand_head in tqdm(range(nentity)):
            if str(rand_head) in ent_conc:
                set_head2conc = set(ent_conc[str(rand_head)])
            else:
                set_head2conc = set_rel2h

            if (rand_head, relation, tail) not in all_triples:
                if len(set_rel2h & set_head2conc) > 0:
                    tmp.append((0, rand_head))
                else:
                    tmp.append((-1, head))
            else:
                tmp.append((-1, head))
        tmp[head] = (0, head)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))
        res.append((positive_sample, negative_sample, filter_bias))

    return res


def _process_tail(triples, pid,rel_t,nentity,ent_conc,all_triples):
    res = []
    tmp = []
    for trip in tqdm(triples, desc="PID:" + str(pid)):
        head, relation, tail = trip
        set_rel2t = set(rel_t[str(relation)])
        for rand_tail in tqdm(range(nentity)):
            if str(rand_tail) in ent_conc:
                set_tail2conc = set(ent_conc[str(rand_tail)])
            else:
                set_tail2conc = set_rel2t
            if (head, relation, rand_tail) not in all_triples:
                if len(set_rel2t & set_tail2conc) > 0:
                    tmp.append((0, rand_tail))
                else:
                    tmp.append((-1, tail))
            else:
                tmp.append((-1, tail))
        tmp[tail] = (0, tail)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))
        res.append((positive_sample, negative_sample, filter_bias))
    return res


if __name__ == '__main__':
    # process to get negative samples
    save_path = 'models/TransE_FB15k-237_concept_domain'
    data_path = 'data_concept/FB15k-237_concept/'

    with open(os.path.join(save_path, 'entities.dict'), 'r') as f:
        entity2id = json.load(f)
    with open(os.path.join(save_path, 'relations.dict'), 'r') as f:
        relation2id = json.load(f)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    with open(os.path.join(data_path, 'rel2dom_h.json')) as fin:
        rel2dom_h = json.load(fin)
    with open(os.path.join(data_path, 'rel2dom_t.json')) as fin:
        rel2dom_t = json.load(fin)
    with open(os.path.join(data_path, 'rel2nn.json')) as fin:
        rel2nn = json.load(fin)
    with open(os.path.join(data_path, 'dom_ent.json')) as fin:
        dom_ent = json.load(fin)
    with open(os.path.join(data_path, 'ent_dom.json')) as fin:
        ent_dom = json.load(fin)

    rel_h = rel2dom_h
    rel_t = rel2dom_t
    rel2nn = rel2nn
    ent_conc = ent_dom
    conc_ents = dom_ent
    negative_size = 128

    with open(os.path.join(save_path, 'train_triples.pkl'), 'rb') as f:
        train_triples = pickle.load(f)
    with open(os.path.join(save_path, 'valid_triples.pkl'), 'rb') as f:
        valid_triples = pickle.load(f)
    with open(os.path.join(save_path, 'test_triples.pkl'), 'rb') as f:
        test_triples = pickle.load(f)

    all_triples = train_triples + valid_triples + test_triples

    count = count_frequency(valid_triples)

    true_head, true_tail = get_true_head_and_tail(valid_triples)

    # multi_work = Parallel(n_jobs=-1, backend='multiprocessing')
    # tasklist = []
    # for i in range(0, 10):
    #     # triples, pid,rel_h,nentity,ent_conc,all_triples
    #     tasklist.append([_process_tail, [], {'triples': test_triples[i * 100: i *100 + 100], 'pid': i, 'rel_t':rel_t,'nentity':nentity,'ent_conc':ent_conc,'all_triples':all_triples}])
    # # task_list = [it.values() for it in taskdic.items()]
    # result = multi_work(tasklist)
    #
    # with open(os.path.join(save_path, 'tail_test_p_n_triples_0_1000.pkl'), 'wb') as f:
    #     pickle.dump(result, f)

    res = []
    for trip in tqdm(valid_triples):
        head, relation, tail = trip
        positive_sample = trip
        subsampling_weight = count[(head, relation)] + count[(tail, -relation-1)]
        negative_sample_list = []
        negative_sample_size = 128
        #tail, relation, rel_t, rel2nn, ent_conc
        e_filter = concept_filter_t(tail, relation, rel_t=rel2dom_t, rel2nn=rel2nn, ent_conc=ent_conc)
        if len(e_filter) > 0:
            ns_size = min(negative_sample_size, len(e_filter))
            negative_sample = np.random.choice(e_filter, ns_size)
            mask = np.in1d(
                negative_sample,
                true_tail[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size = negative_sample.size

        while negative_sample_size < 128:
            negative_sample = np.random.randint(nentity, size=128)
            mask = np.in1d(
                negative_sample,
                true_tail[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:128]

        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
        res.append((positive_sample, negative_sample, subsampling_weight))

    with open(os.path.join(save_path, 'valid_tail_triples.pkl'), 'wb') as f:
        pickle.dump(res, f)
