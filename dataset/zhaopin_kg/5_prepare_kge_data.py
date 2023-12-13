import os
import dgl
import numpy as np

data_path = 'kg/'

# load dgl graph from directory
kg = dgl.data.CSVDataset(data_path, force_reload=True)[0]

# prepare data for KGE
kge_dir = 'kge/train_data/'
if not os.path.exists(kge_dir):
    os.makedirs(kge_dir)
train_out = []
valid_out = []
test_out = []
for u,r,v in kg.canonical_etypes:
    src_ids, dst_ids = kg.edges(etype=r)
    # generate a random permutation of the nodes
    perm = np.random.permutation(len(src_ids))
    # reoder the nodes
    src_ids = src_ids[perm]
    dst_ids = dst_ids[perm]
    # 98% train, 1% valid, 1% test
    train_size = int(len(src_ids) * 0.8)
    valid_size = int(len(src_ids) * 0.1)
    test_size = int(len(src_ids) * 0.1)
    for src_id, dst_id in zip(src_ids[:train_size], dst_ids[:train_size]):
        line = f"{u}ID{src_id.item()}\t{r}\t{v}ID{dst_id.item()}\n"
        train_out.append(line)
    for src_id, dst_id in zip(src_ids[train_size:train_size+valid_size], dst_ids[train_size:train_size+valid_size]):
        line = f"{u}ID{src_id.item()}\t{r}\t{v}ID{dst_id.item()}\n"
        valid_out.append(line)
    for src_id, dst_id in zip(src_ids[train_size+valid_size:], dst_ids[train_size+valid_size:]):
        line = f"{u}ID{src_id.item()}\t{r}\t{v}ID{dst_id.item()}\n"
        test_out.append(line)

# shuffle
np.random.shuffle(train_out)
np.random.shuffle(valid_out)
np.random.shuffle(test_out)

# remove last '\n'
train_out[-1] = train_out[-1][:-1]
valid_out[-1] = valid_out[-1][:-1]
test_out[-1] = test_out[-1][:-1]

with open(os.path.join(kge_dir, 'train.txt'), 'w') as f:
    f.writelines(train_out)
with open(os.path.join(kge_dir, 'valid.txt'), 'w') as f:
    f.writelines(valid_out)
with open(os.path.join(kge_dir, 'test.txt'), 'w') as f:
    f.writelines(test_out)