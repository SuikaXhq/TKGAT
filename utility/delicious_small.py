import os
import numpy as np
# from math import ceil

dataset_dir = './datasets/hetrec2011-delicious-2k'
interaction_file_name = 'user_taggedbookmarks.dat'
interaction_file_name_small = 'user_taggedbookmarks_small.dat'
interaction_file_path = os.path.join(dataset_dir, interaction_file_name)
interaction_file_path_small = os.path.join(dataset_dir, interaction_file_name_small)

tag_file_name = 'tags.dat'
tag_file_name_small = 'tags_small.dat'
tag_file_path = os.path.join(dataset_dir, tag_file_name)
tag_file_path_small = os.path.join(dataset_dir, tag_file_name_small)

data = np.loadtxt(interaction_file_path, dtype=np.str_, skiprows=1, delimiter='\t')
tags = np.loadtxt(tag_file_path, dtype=np.str_, skiprows=1, delimiter='\t')
# remain_len = ceil(data.shape[0] * 0.2)
# remain_idx = np.random.choice(np.arange(data.shape[0]), remain_len, replace=False)
# remain_idx.sort()
# data = data[remain_idx, :]

hop = 15 # remove tags that appears < hop times

unique, counts = np.unique(data[:, 2], return_counts=True)
remain_indices = unique[counts >= hop]
remain_mask = np.isin(data[:, 2], remain_indices)
remain_mask_tags = np.isin(tags[:, 0], remain_indices)
data = data[remain_mask, :]
tags = tags[remain_mask_tags, :]

np.savetxt(interaction_file_path_small, data, fmt='%s', delimiter='\t', header='userID\tbookmarkID\ttagID\tday\tmonth\tyear\thour\tminute\tsecond\n')
np.savetxt(tag_file_path_small, tags, fmt='%s', delimiter='\t', header='id\tvalue\n')