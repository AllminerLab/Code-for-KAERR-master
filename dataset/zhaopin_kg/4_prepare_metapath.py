import os
import dgl
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import json
import pickle

data_path = 'kg/'
meta_path_dir = 'kg/meta_path/'
if not os.path.exists(meta_path_dir):
    os.makedirs(meta_path_dir)

# load dgl graph from directory
kg = dgl.data.CSVDataset(data_path, force_reload=True)[0]
kg = kg.to('cuda:3')
num_user = kg.number_of_nodes('user')
num_item = kg.number_of_nodes('job')
print('num_user: ', num_user)
print('num_item: ', num_item)

# define the meta-path
meta_paths = [
    ['cur_city', 'job_city_rev'],
    ['desire_city', 'job_city_rev'],

    ['cur_min_salary', 'min_salary_rev'],
    ['cur_min_salary', 'higher_salary', 'min_salary_rev'],
    ['cur_min_salary', 'lower_salary', 'min_salary_rev'],

    ['cur_min_salary', 'max_salary_rev'],
    ['cur_min_salary', 'higher_salary', 'max_salary_rev'],
    ['cur_min_salary', 'lower_salary', 'max_salary_rev'],

    ['cur_max_salary', 'min_salary_rev'],
    ['cur_max_salary', 'higher_salary', 'min_salary_rev'],
    ['cur_max_salary', 'lower_salary', 'min_salary_rev'],

    ['cur_max_salary', 'max_salary_rev'],
    ['cur_max_salary', 'higher_salary', 'max_salary_rev'],
    ['cur_max_salary', 'lower_salary', 'max_salary_rev'],

    ['desire_min_salary', 'min_salary_rev'],
    ['desire_min_salary', 'higher_salary', 'min_salary_rev'],
    ['desire_min_salary', 'lower_salary', 'min_salary_rev'],

    ['desire_min_salary', 'max_salary_rev'],
    ['desire_min_salary', 'higher_salary', 'max_salary_rev'],
    ['desire_min_salary', 'lower_salary', 'max_salary_rev'],

    ['desire_max_salary', 'min_salary_rev'],
    ['desire_max_salary', 'higher_salary', 'min_salary_rev'],
    ['desire_max_salary', 'lower_salary', 'min_salary_rev'],

    ['desire_max_salary', 'max_salary_rev'],
    ['desire_max_salary', 'higher_salary', 'max_salary_rev'],
    ['desire_max_salary', 'lower_salary', 'max_salary_rev'],
    
    ['cur_jdtype', 'job_jdtype_rev'],
    ['desire_jdtype', 'job_jdtype_rev'],

    ['cur_degree', 'require_degree_rev'],
    ['cur_degree', 'higher_degree', 'require_degree_rev'],
    ['cur_degree', 'lower_degree', 'require_degree_rev'],

    ['cur_year', 'require_year_rev'],
    ['cur_year', 'higher_year', 'require_year_rev'],
    ['cur_year', 'lower_year', 'require_year_rev'],
    
    ['cur_experience', 'require_experience_rev'],
    ['cur_experience', 'similar_experience', 'require_experience_rev'],
]
json.dump(meta_paths, open(os.path.join(meta_path_dir, 'meta_path.json'), 'w'))

type_start_index = {kg.ntypes[0]: 0}
for i, ntype in enumerate(kg.ntypes[1:]):
    type_start_index[ntype] = type_start_index[kg.ntypes[i]] + kg.number_of_nodes(kg.ntypes[i])

def sample_metapath(kg, metapath, start_node, num_path):
    traces, eids, types = dgl.sampling.random_walk(
        kg, [start_node]*num_path, metapath=metapath, return_eids=True)
    # remove the trace where last node is -1
    keep_idx = (traces[:, -1] != -1).nonzero(as_tuple=True)[0]
    traces = traces[keep_idx]
    eids = eids[keep_idx]
    paths = torch.cat([traces, eids], dim=-1)
    paths = torch.unique(paths, dim=0)
    out_paths = torch.zeros_like(paths)
    out_paths[:, 0::2] = paths[:, :paths.shape[-1]//2+1]
    out_paths[:, 1::2] = paths[:, paths.shape[-1]//2+1:]
    return out_paths

all_paths = []
path_lens = []
ui_path_index = [[] for _ in range(num_user*num_item)]
max_path_len = max([len(meta_path)*2+1 for meta_path in meta_paths])

cur_start_index = 0
for meta_path in meta_paths:
    print("processing meta-path: ", meta_path)
    max_try = 10
    if len(meta_path) > 2:
        num_path = 2000000
    if meta_path[-1].split('_')[-1] == 'experience':
        num_path = 2000000
    else:
        num_path = 1000000
    user_path = []
    # 1. Use metapath_reachable_graph to check if the user and job are connected
    uj_graph = dgl.metapath_reachable_graph(kg, meta_path)
    for user_index in tqdm(range(kg.number_of_nodes('user'))):
        # 后继节点
        job_indexs = uj_graph.successors(user_index)
        if job_indexs.shape[0] == 0:
            continue
        paths = sample_metapath(kg, meta_path, user_index, num_path)
        sampled_job = paths[:, -1].unique().shape[0]
        count = 0
        while sampled_job < job_indexs.shape[0] and count < max_try:
            paths = torch.cat([paths, sample_metapath(kg, meta_path, user_index, num_path)], dim=0)
            paths = torch.unique(paths, dim=0)
            sampled_job = paths[:, -1].unique().shape[0]
            count += 1
        for i in range(paths.shape[0]):
            uid = paths[i, 0].item()
            jid = paths[i, -1].item()
            uj_pair_index = uid * num_item + jid
            ui_path_index[uj_pair_index].append(cur_start_index + i)
            
        for i, r_type in enumerate(meta_path):
            r_index = kg.etypes.index(r_type)
            paths[:, 2*i+1] = r_index + kg.num_nodes() + 1
            src_type, dst_type = [(triple[0], triple[2]) for triple in kg.canonical_etypes if triple[1] == r_type][0]
            paths[:, 2*i] = paths[:, 2*i] + type_start_index[src_type] + 1
        paths[:, -1] = paths[:, -1] + type_start_index[dst_type] + 1
        paths = paths.cpu().numpy()
        user_path.append(paths)
        cur_start_index += paths.shape[0]
    
    user_path = np.concatenate(user_path, axis=0)
    if user_path.shape[1] < max_path_len:
        user_path = np.concatenate([user_path, np.zeros((user_path.shape[0], max_path_len-user_path.shape[1]))], axis=-1)
    user_path = user_path.astype(np.int32)
    path_len = np.ones(user_path.shape[0]) * (len(meta_path) * 2 + 1)
    path_lens.append(path_len)
    #np.save(meta_path_dir + ','.join(meta_path) + '.npy', user_path)
    all_paths.append(user_path)

all_paths = np.concatenate(all_paths, axis=0)
path_len = np.concatenate(path_lens, axis=0)
path_len = path_len.astype(np.int32)

np.save(os.path.join(meta_path_dir, 'path_len.npy'), path_len)

ui_path_index_out = np.zeros((len(ui_path_index), 32), dtype=np.int32)
for i, index in enumerate(ui_path_index):
    index = index[:32]
    ui_path_index_out[i, :len(index)] = index

np.save(os.path.join(meta_path_dir, 'ui_path_index.npy'), ui_path_index_out)


np.save(os.path.join(meta_path_dir, 'all_paths.npy'), all_paths)

    