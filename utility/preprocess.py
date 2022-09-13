import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import collections
import numpy as np
from math import ceil

data_dir = './datasets'
glove_file_name = 'glove.6B.100d.txt'
word2vec_file_name = 'glove.6B.100d.w2vformat.txt'
glove_path = os.path.join(data_dir, glove_file_name)
word2vec_path = os.path.join(data_dir, word2vec_file_name)
word2vec_dump_path = os.path.join(data_dir, 'pretrain/GloVe.kv')

dataset_name = 'delicious_small'

if dataset_name == 'movielens':
    dataset_dir = './datasets/hetrec2011-movielens-2k-v2'
    interaction_file_name = 'user_ratedmovies.dat'
elif dataset_name == 'lastfm':
    dataset_dir = './datasets/hetrec2011-lastfm-2k'
    interaction_file_name = 'user_artists.dat'
elif dataset_name == 'delicious':
    dataset_dir = './datasets/hetrec2011-delicious-2k'
    interaction_file_name = 'user_taggedbookmarks.dat'
elif dataset_name == 'delicious_small':
    dataset_dir = './datasets/hetrec2011-delicious-2k'
    interaction_file_name = 'user_taggedbookmarks_small.dat'
interaction_file_path = os.path.join(dataset_dir, interaction_file_name)

np.random.seed(123)

def read_glove(file_path, save_path=None, format='w2v'):
    r'''Read pretrained GloVe word vectors from txt file. The format ('GloVe' or 'word2vec') should be specified by user.
    Parameters:
    -----
    format - enumerate from {'w2v', 'glove'}, if 'glove', fucntion will automatically convert to w2v format and then read it.

    Return:
    KeyedVectors
    The word vectors read from file.
    -----

    '''
    # process format
    if format == 'w2v':
        w2v_path = file_path
    elif format == 'glove':
        data_dir, file_name = os.path.split(file_path)
        fname, ext = os.path.splitext(file_name)
        w2v_path = os.path.join(data_dir, fname + '.w2vformat.' + ext)
        if os.path.exists(file_path):
            glove2word2vec(file_path, w2v_path)
            print('Converted txt file save at {}'.format(word2vec_path))
        else:
            raise FileNotFoundError('File {} is not found.'.format(glove_path))
    else:
        raise NotImplementedError("Parameter format need to be either 'w2v' or 'glove'.")
    
    # read vectors
    wordvec = KeyedVectors.load_word2vec_format(w2v_path, binary=False, datatype=np.float32)

    # save keyed vector
    if save_path is not None:
        wordvec.save(save_path)

    return wordvec

def initialize_embeddings(n_entities, embed_dim, word2vec, word_lists):
    r'''Initialize node embeddings using word vectors.
    Parameters:
    -----
    n_entities - Number of entities that need to be initialized.
    embed_dim - Embedding vector dimensionality.
    word2vec - The word2vec embeddings, a gensim.models.KeyedVectors instance.
    word_lists - A entity-text list for looking up.

    Return:
    -----

    '''
    # construct embeddings
    
def construct_interaction_dataset(dataset_name, interaction_path, core_number=5, train_test_ratio=[0.7, 0.1, 0.2]):
    r'''Construct user-item graph, saved as txt in data_dir/dataset_name/.
    Parameters:
    -----
    dataset_name - Dataset folder name. data_dir/dataset_name should not exist.
    interaction_path - Original interaction list, contains original user id and item id at the 1st, 2nd tokens in each line.
    core_number - Number of the least interactions. Default 5.
    train_test_ratio - List of train, valid, test ratio, default [0.7, 0.1, 0.2].

    Return:
    -----
    No returns.
    '''
    # read user, item, interaction list files
    dataset_path = os.path.join(data_dir, dataset_name)
    if os.path.exists(dataset_path):
        print('Data folder {} exists. Try another dataset name.'.format(dataset_path))
        return
    else:
        os.mkdir(dataset_path)
    with open(interaction_path, mode='r') as interaction_file:
        interaction_list = interaction_file.readlines()[1:] # first line is the column names, discard
    print('Read Interaction file done, processing...')

    # reconstruct user and item id, aligned as int
    interactions_org = collections.defaultdict(list)
    interactions = {}
    for line in interaction_list:
        user, item = int(line.split()[0]), int(line.split()[1])
        if item not in interactions_org[user]:
            interactions_org[user].append(item)
    users_org = list(interactions_org.keys())
    
    # select users that interact with more than given number (parameter 'core_number') of items
    for user in users_org:
        if len(interactions_org[user]) < core_number:
            interactions_org.pop(user)
    users = sorted(list(interactions_org.keys()))
    items = sorted(list(set([item for v in interactions_org.values() for item in v])))
    for user_id, user in enumerate(users):
        interactions[user_id] = [items.index(item) for item in interactions_org.pop(user)]

    # save lists
    with open(os.path.join(dataset_path, 'user_list.txt'), mode='w') as user_reconstruct_file:
        user_reconstruct_file.write('org_id remap_id\n')
        for user_id, org_user_id in enumerate(users):
            user_reconstruct_file.write(str(org_user_id) + ' ' + str(user_id) + '\n')
    print('Saved user list.')
    with open(os.path.join(dataset_path, 'item_list.txt'), mode='w') as item_reconstruct_file:
        item_reconstruct_file.write('org_id remap_id\n')
        for item_id, org_item_id in enumerate(items):
            item_reconstruct_file.write(str(org_item_id) + ' ' + str(item_id) + '\n')
    print('Saved item list.')
    with open(os.path.join(dataset_path, 'interactions.txt'), mode='w') as interaction_reconstruct_file:
        for user, user_interactions in interactions.items():
            interaction_reconstruct_file.write(str(user) + ' ' + ' '.join(map(str, user_interactions)) + '\n')
    print('Saved interaction list.')

    # train-test split, each user have at least 1 test sample and 1 valid sample
    train_list = collections.defaultdict(list)
    valid_list = collections.defaultdict(list)
    test_list = collections.defaultdict(list)
    for user_id, user_interactions in interactions.items():
        length = len(user_interactions)
        user_interactions = np.asarray(user_interactions)
        indices = np.arange(length, dtype=np.int32)
        np.random.shuffle(indices)
        test_length = ceil(length * train_test_ratio[2])
        valid_length = ceil(length * train_test_ratio[1]) + test_length
        test_list[user_id] = user_interactions[indices[:test_length]]
        valid_list[user_id] = user_interactions[indices[test_length:valid_length]]
        train_list[user_id] = user_interactions[indices[valid_length:]]
    train_file = open(os.path.join(dataset_path, 'train.txt'), mode='w')
    valid_file = open(os.path.join(dataset_path, 'valid.txt'), mode='w')
    test_file = open(os.path.join(dataset_path, 'test.txt'), mode='w')
    for user in range(len(users)):
        train_file.write(str(user) + ' ' + ' '.join(map(str, train_list[user])) + '\n')
        valid_file.write(str(user) + ' ' + ' '.join(map(str, valid_list[user])) + '\n')
        test_file.write(str(user) + ' ' + ' '.join(map(str, test_list[user])) + '\n')
    train_file.close()
    valid_file.close()
    test_file.close()
    print('Saved train-valid-test list.')

def load_raw_kg(file_path, multiple_in_one_line=False, column_sep='\t'):
    r'''Load relation from file.

    Parameters:
    -----
    - file_path - str, relation file path.
    - multiple_in_one_line - bool, tail entities are listed in one line or not, default False.
    - column_sep - column separate token, default '\t'.

    Return:
    -----
    - defaultdict(list) - relation dictionary, head as key and tails saved in value list.
    '''
    with open(file_path, mode='r') as f:
        lines = f.readlines()[1:] # discard column names
    relation_dict = collections.defaultdict(list)
    for line in lines:
        items = line.split(column_sep)
        if multiple_in_one_line:
            head, tails = items[0].strip(), [item.strip() for item in items[1:]]
            relation_dict[head].extend(tails)
        else:
            head, tail = items[0].strip(), items[1].strip()
            relation_dict[head].append(tail)
    return relation_dict

def extract_entity(relation_dict):
    r'''Extract entities from relation dictionary (obtained by load_raw_kg)
    
    Parameters:
    -----
    - relation_dict - See load_raw_kg(...)

    Return:
    -----
    - list - entity list
    '''
    entity_set = set()
    for entities in relation_dict.values():
        for entity in entities:
            entity_set.add(entity)
    entity_list = list(entity_set)
    return entity_list


def construct_kg(dataset, raw_dataset_dir):
    r'''Construct knowledge graph
    Parameters:
    -----
    dataset - dataset name.\\
    dataset_dir - raw dataset directory path.

    Return:
    -----
    No returns.
    '''
    if dataset == 'movielens':
        # process HetRec2011 MovieLens Dataset
        print('Processing {}...'.format(dataset))

        # read user/item lists
        save_dir = os.path.join(data_dir, dataset)
        with open(os.path.join(save_dir, 'user_list.txt'), mode='r') as f:
            user_list_lines = f.readlines()[1:] # first line is the column names, discard
        with open(os.path.join(save_dir, 'item_list.txt'), mode='r') as f:
            item_list_lines = f.readlines()[1:] # first line is the column names, discard

        user_list = [None]*len(user_list_lines)
        item_list = [None]*len(item_list_lines)
        for line in user_list_lines:
            org_id, index = line.split()
            index = int(index)
            user_list[index] = org_id
        for line in item_list_lines:
            org_id, index = line.split()
            index = int(index)
            item_list[index] = org_id

        # map original ID into remap ID
        user_id_map = {user: id for id, user in enumerate(user_list)}
        item_id_map = {item: id for id, item in enumerate(item_list)}
        print('User/Item ID map loaded.')

        # read genres, directors, actors, countries, locations
        genre_dict = load_raw_kg(os.path.join(raw_dataset_dir, 'movie_genres.dat'))
        director_dict = load_raw_kg(os.path.join(raw_dataset_dir, 'movie_directors.dat'))
        actor_dict = load_raw_kg(os.path.join(raw_dataset_dir, 'movie_actors.dat'))
        country_dict = load_raw_kg(os.path.join(raw_dataset_dir, 'movie_countries.dat'))
        location_dict = load_raw_kg(os.path.join(raw_dataset_dir, 'movie_locations.dat'), multiple_in_one_line=True)

        raw_relation_dicts = [
            genre_dict,
            director_dict,
            actor_dict,
            country_dict,
            location_dict
        ]

        relation_list = [
            'genre',
            'director',
            'actor',
            'country',
            'location'
        ]
        for relation in relation_list:
            print('Loaded relation {}.'.format(relation))

        genre_list = extract_entity(genre_dict)
        director_list = extract_entity(director_dict)
        actor_list = extract_entity(actor_dict)
        country_list = extract_entity(country_dict)
        location_list = extract_entity(location_dict)

        # put items in entity list
        entity_list = \
            item_list + \
            genre_list + \
            director_list + \
            actor_list + \
            country_list + \
            location_list
        entity_list = sorted(set(entity_list), key = entity_list.index)
        print('Entity list constructed.')

        # remap entity ids
        entity_id_map = {}
        for id, entity in enumerate(entity_list):
            entity_id_map[entity] = id

        # construct and save kg
        kg = collections.defaultdict(list)
        for item in item_list:
            for i, relation_dict in enumerate(raw_relation_dicts):
                kg[(item_id_map[item], i)].extend([entity_id_map[entity] for entity in relation_dict[item]])
        with open(os.path.join(save_dir, 'entity_list.txt'), mode='w') as f:
            f.write('org_id\tremap_id\n')
            for id, entity in enumerate(entity_list):
                f.write('{}\t{}\n'.format(entity, id))
        with open(os.path.join(save_dir, 'relation_list.txt'), mode='w') as f:
            f.write('org_id\tremap_id\n')
            for id, relation in enumerate(relation_list):
                f.write('{}\t{}\n'.format(relation, id))
        with open(os.path.join(save_dir, 'kg_final.txt'), mode='w') as f:
            for (head, relation), tails in kg.items():
                for tail in tails:
                    f.write('{} {} {}\n'.format(head, relation, tail))
        print('KG saved.')

        # read tags
        tag_list = []
        with open(os.path.join(raw_dataset_dir, 'tags.dat'), mode='r') as f:
            for line in f.readlines()[1:]:
                tag_id, description = line.split('\t')
                tag_list.append((tag_id, description.strip()))
        tag_id_map = {item[0]: id for id, item in enumerate(tag_list)}
        
        tag_interaction_list = []
        with open(os.path.join(raw_dataset_dir, 'user_taggedmovies.dat'), mode='r') as f:
            for line in f.readlines()[1:]:
                user, movie, tag = line.split('\t')[:3]
                if user in user_id_map and tag in tag_id_map and movie in item_id_map:
                    tag_interaction_list.append((
                        user_id_map[user],
                        tag_id_map[tag],
                        item_id_map[movie]
                    ))
            
        # save tag interactions
        with open(os.path.join(save_dir, 'tag_list.txt'), mode='w') as f:
            f.write('org_id\tremap_id\tdescription\n')
            for id, (tag, description) in enumerate(tag_list):
                f.write('{}\t{}\t{}\n'.format(tag, id, description))
        with open(os.path.join(save_dir, 'tagging.txt'), mode='w') as f:
            for user, tag, item in tag_interaction_list:
                f.write('{} {} {}\n'.format(user, tag, item))
        print('Tag interactions saved.')

    elif dataset == 'lastfm':
        # process HetRec2011 LastFM Dataset
        print('Processing {}...'.format(dataset))

        # read user/item lists
        save_dir = os.path.join(data_dir, dataset)
        with open(os.path.join(save_dir, 'user_list.txt'), mode='r') as f:
            user_list_lines = f.readlines()[1:] # first line is the column names, discard
        with open(os.path.join(save_dir, 'item_list.txt'), mode='r') as f:
            item_list_lines = f.readlines()[1:] # first line is the column names, discard

        user_list = [None]*len(user_list_lines)
        item_list = [None]*len(item_list_lines)
        for line in user_list_lines:
            org_id, index = line.split()
            index = int(index)
            user_list[index] = org_id
        for line in item_list_lines:
            org_id, index = line.split()
            index = int(index)
            item_list[index] = org_id

        # map original ID into remap ID
        user_id_map = {user: id for id, user in enumerate(user_list)}
        item_id_map = {item: id for id, item in enumerate(item_list)}
        print('User/Item ID map loaded.')

        # read social network
        social_dict = load_raw_kg(os.path.join(raw_dataset_dir, 'user_friends.dat'))

        # remap user ids
        user_id_map = {}
        for id, user in enumerate(user_list):
            user_id_map[user] = id

        # construct and save social network (sn)
        sn = collections.defaultdict(list)
        for user, friends in social_dict.items():
            if user in user_id_map:
                for friend in friends:
                    if friend in user_id_map:
                        sn[user_id_map[user]].append(user_id_map[friend])
        with open(os.path.join(save_dir, 'social_network.txt'), mode='w') as f:
            for head, tails in sn.items():
                for tail in tails:
                    f.write('{} 0 {}\n'.format(head, tail))
        print('Social network saved.')

        # read tags
        tag_list = []
        with open(os.path.join(raw_dataset_dir, 'tags.dat'), mode='r') as f:
            for line in f.readlines()[1:]:
                tag_id, description = line.split('\t')
                tag_list.append((tag_id, description.strip()))
        tag_id_map = {item[0]: id for id, item in enumerate(tag_list)}
        
        tag_interaction_list = []
        with open(os.path.join(raw_dataset_dir, 'user_taggedartists.dat'), mode='r') as f:
            for line in f.readlines()[1:]:
                user, movie, tag = line.split('\t')[:3]
                if user in user_id_map and tag in tag_id_map and movie in item_id_map:
                    tag_interaction_list.append((
                        user_id_map[user],
                        tag_id_map[tag],
                        item_id_map[movie]
                    ))
            
        # save tag interactions
        with open(os.path.join(save_dir, 'tag_list.txt'), mode='w') as f:
            f.write('org_id\tremap_id\tdescription\n')
            for id, (tag, description) in enumerate(tag_list):
                f.write('{}\t{}\t{}\n'.format(tag, id, description))
        with open(os.path.join(save_dir, 'tagging.txt'), mode='w') as f:
            for user, tag, item in tag_interaction_list:
                f.write('{} {} {}\n'.format(user, tag, item))
        print('Tag interactions saved.')

    elif dataset == 'delicious' or dataset == 'delicious_small':
        # process HetRec2011 Delicious Dataset
        print('Processing {}...'.format(dataset))

        # read user/item lists
        save_dir = os.path.join(data_dir, dataset)
        with open(os.path.join(save_dir, 'user_list.txt'), mode='r') as f:
            user_list_lines = f.readlines()[1:] # first line is the column names, discard
        with open(os.path.join(save_dir, 'item_list.txt'), mode='r') as f:
            item_list_lines = f.readlines()[1:] # first line is the column names, discard

        user_list = [None]*len(user_list_lines)
        item_list = [None]*len(item_list_lines)
        for line in user_list_lines:
            org_id, index = line.split()
            index = int(index)
            user_list[index] = org_id
        for line in item_list_lines:
            org_id, index = line.split()
            index = int(index)
            item_list[index] = org_id

        # map original ID into remap ID
        user_id_map = {user: id for id, user in enumerate(user_list)}
        item_id_map = {item: id for id, item in enumerate(item_list)}
        print('User/Item ID map loaded.')

        # read social network
        social_dict = load_raw_kg(os.path.join(raw_dataset_dir, 'user_contacts.dat'))

        # remap user ids
        user_id_map = {}
        for id, user in enumerate(user_list):
            user_id_map[user] = id

        # construct and save social network (sn)
        sn = collections.defaultdict(list)
        for user, friends in social_dict.items():
            if user in user_id_map:
                for friend in friends:
                    if friend in user_id_map:
                        sn[user_id_map[user]].append(user_id_map[friend])
        with open(os.path.join(save_dir, 'social_network.txt'), mode='w') as f:
            for head, tails in sn.items():
                for tail in tails:
                    f.write('{} 0 {}\n'.format(head, tail))
        print('Social network saved.')

        # read tags
        tag_list = []
        with open(os.path.join(raw_dataset_dir, 'tags.dat' if dataset == 'delicious' else 'tags_small.dat'), mode='r') as f:
            for line in f.readlines()[1:]:
                tag_id, description = line.split('\t')
                tag_list.append((tag_id, description.strip()))
        tag_id_map = {item[0]: id for id, item in enumerate(tag_list)}
        
        tag_interaction_list = []
        with open(os.path.join(raw_dataset_dir, 'user_taggedbookmarks.dat' if dataset == 'delicious' else 'user_taggedbookmarks_small.dat'), mode='r') as f:
            for line in f.readlines()[1:]:
                user, movie, tag = line.split('\t')[:3]
                if user in user_id_map and tag in tag_id_map and movie in item_id_map:
                    tag_interaction_list.append((
                        user_id_map[user],
                        tag_id_map[tag],
                        item_id_map[movie]
                    ))
            
        # save tag interactions
        with open(os.path.join(save_dir, 'tag_list.txt'), mode='w') as f:
            f.write('org_id\tremap_id\tdescription\n')
            for id, (tag, description) in enumerate(tag_list):
                f.write('{}\t{}\t{}\n'.format(tag, id, description))
        with open(os.path.join(save_dir, 'tagging.txt'), mode='w') as f:
            for user, tag, item in tag_interaction_list:
                f.write('{} {} {}\n'.format(user, tag, item))
        print('Tag interactions saved.')

    else:
        raise NotImplementedError('KG construction of dataset \'{}\' is not implemented.'.format(dataset))

if __name__ == '__main__':
    construct_interaction_dataset(dataset_name, interaction_file_path)
    construct_kg(dataset_name, dataset_dir)
    
    # read_glove(word2vec_path, save_path=word2vec_dump_path, format='w2v')