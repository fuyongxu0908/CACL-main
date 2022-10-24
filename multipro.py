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
    with open(os.path.join(data_path, 'rel2nn.json')) as fin:
        rel2nn = json.load(fin)
    with open(os.path.join(data_path, 'dom_ent.json')) as fin:
        dom_ent = json.load(fin)
    with open(os.path.join(data_path, 'ent_dom.json')) as fin:
        ent_dom = json.load(fin)

    rel_h = rel2dom_h
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

    count = count_frequency(test_triples)

    true_head, true_tail = get_true_head_and_tail(test_triples)


    multi_work = Parallel(n_jobs=-1, backend='multiprocessing')
    tasklist = []
    for i in range(0, 6):
        # triples, pid,rel_h,nentity,ent_conc,all_triples
        tasklist.append([_process, [], {'triples': test_triples[20400 + i *11: 20400 + i *11 + 11], 'pid': i, 'rel_h':rel_h,'nentity':nentity,'ent_conc':ent_conc,'all_triples':all_triples}])
    # task_list = [it.values() for it in taskdic.items()]
    result = multi_work(tasklist)

    with open(os.path.join(save_path, 'testtest_p_n_triples_final.pkl'), 'wb') as f:
        pickle.dump(result, f)

#