import dill
from collections import Counter, defaultdict
import bidict
from tqdm import tqdm
import torch

from .utils import *
from .calc import *
from .fragment_bond import FragmentBond, FragBondList
from .fragment import Fragment
from .fragment_group import FragmentGroup
from .fragmentizer import Fragmentizer
from .fragment_tree import FragmentTree


class Vocab:
    BOS = '<BOS>'
    EOS = '<EOS>'
    PAD = '<PAD>'
    UNK = '<UNK>'
    
    TOKENS = [BOS, EOS, PAD, UNK]

    

    def __init__(self, monoatomic_tokens_path, smiles_counter_path, joint_counter_path, threshold, save_path=None):
        self.threshold = threshold
        if monoatomic_tokens_path is not None and joint_counter_path is not None:
            # load monoatomic tokens and fragment counter
            print(f"Loading fragment library", end='...')
            mono_bond_potential = defaultdict(int)
            with open(monoatomic_tokens_path, 'r') as f:
                for line in f:
                    frag = Fragment.parse_fragment_string(line.strip())
                    joint = frag_to_joint_list(tuple(frag))
                    if len(joint) > 0:
                        joint = joint[0]
                        potential = joint[1] + joint[2] * 2 + joint[3] * 3
                        mono_bond_potential[frag.smiles] = potential
                    else:
                        mono_bond_potential[frag.smiles] = 0
            mono_bond_potential = dict(mono_bond_potential)

            if smiles_counter_path.endswith('.pkl'):
                smiles_counter = dill.load(open(smiles_counter_path, 'rb'))
                smiles_list = [smiles for smiles, cnt in smiles_counter.items() if cnt >= threshold]
            else:
                with open(smiles_counter_path, 'r') as f:
                    smiles_list = [line.strip().split('\t')[0] for line in f if int(line.strip().split('\t')[1]) >= threshold]
            
            if joint_counter_path.endswith('.pkl'):
                joint_counter = dill.load(open(joint_counter_path, 'rb'))
            else:
                with open(joint_counter_path, 'r') as f:
                    joint_counter = {line.split('\t')[0]: {eval(joint):cnt for joint, cnt in zip(line.split('\t')[1::2], line.split('\t')[2::2])} for line in f}
            
            bond_poteintials = defaultdict(lambda: defaultdict(int))
            for smi, potential in mono_bond_potential.items():
                bond_poteintials[smi][0] = potential
            for smi in smiles_list:
                for joint, cnt in joint_counter[smi].items():
                    p = joint[1] + joint[2] * 2 + joint[3] * 3
                    if p > bond_poteintials[smi][joint[0]]:
                        bond_poteintials[smi][joint[0]] = p
            print("done")

            # get the vocab
            max_joint_cnt = max([len(v) for v in bond_poteintials.values()])
            self.idx_to_token = bidict({i:v for i, v in enumerate(Vocab.TOKENS)})
            self.idx_to_token.update({i+len(Vocab.TOKENS):smi for i, smi in enumerate(bond_poteintials.keys())})           
            self._set_token_idx()

            joint_potential_tensor = torch.full((len(self), max_joint_cnt, 2), -1, dtype=torch.int32) # (vocab_size, max_joint_cnt, 2(atom_idx, potential))
            for idx, token in tqdm(self.idx_to_token.items(), desc='JointFromVocab'):
                if token in Vocab.TOKENS:
                    continue
                for i, (atom_idx, potential) in enumerate(bond_poteintials[token].items()):
                    joint_potential_tensor[idx, i] = torch.tensor([atom_idx, potential], dtype=torch.int32)
            self.joint_potential_tensor = joint_potential_tensor

            # build fragment tree
            self.tree = FragmentTree()
            self.tree.add_fragment_list(bond_poteintials)

            # calculate smilarity matrix
            fp_tensor = smiles_to_fp_tensor([self[i] for i in range(len(self.TOKENS), len(self))]).to(torch.device('cpu'))
            cosine_matrix = compute_cosine_similarity(fp_tensor)
            self.cosine_matrix = torch.zeros(len(self), len(self), dtype=torch.float32)
            self.cosine_matrix[:len(self.TOKENS), :len(self.TOKENS)] = torch.eye(len(self.TOKENS))
            self.cosine_matrix[len(self.TOKENS):, len(self.TOKENS):] = cosine_matrix
            self.fp_tensor = torch.cat([torch.zeros(len(self.TOKENS),fp_tensor.size(1), dtype=fp_tensor.dtype), fp_tensor], dim=0)

            # calculate tgt cosine matrix
            similarity_weight = torch.full_like(self.cosine_matrix, 0.5)
            torch.diagonal(similarity_weight).fill_(1.0)
            self.tgt_cosine_matrix = self.cosine_matrix * similarity_weight

        self.fragmentizer = Fragmentizer()

        if save_path:
            print(f"Saving vocabulary to {save_path}", end='...')
            self.save(save_path)
            print("done")
            print(f"Vocabulary size (>={self.threshold}): {len(self)}")

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.idx_to_token[key]
        elif isinstance(key, str):
            return self.idx_to_token.inv[key]
    
    def _set_token_idx(self):
        self.bos = self[self.BOS]
        self.eos = self[self.EOS]
        self.pad = self[self.PAD]
        self.unk = self[self.UNK]

    def get_data_to_save(self):
        data_to_save = {
            'token': self.idx_to_token,
            'joint_potential': self.joint_potential_tensor,
            'tree': self.tree.root.to_list(),
            'tree_smi_potential': self.tree.smiles_and_atom_idx_to_potential,
            'fingerprint': self.fp_tensor,
            'cosine_matrix': self.cosine_matrix,
            'tgt_cosine_matrix': self.tgt_cosine_matrix,
            'threshold': self.threshold,
            'bos': self.bos,
            'eos': self.eos,
            'pad': self.pad,
            'unk': self.unk,
        }
        return data_to_save


    def save(self, path):
        data_to_save = self.get_data_to_save()
        dill.dump(data_to_save, open(path, 'wb'))


    @staticmethod
    def get_vocab_from_data(data):
        token_data = data['token']
        joint_potential_data = data['joint_potential']
        tree_data = data['tree']
        tree_smi_potential_data = data['tree_smi_potential']
        fingerprint_data = data['fingerprint']
        cosine_matrix_data = data['cosine_matrix']
        tgt_cosine_matrix = data['tgt_cosine_matrix']
        threshold = data['threshold']
        vocab = Vocab(None, None, None, threshold)
        vocab.idx_to_token = token_data
        vocab._set_token_idx()
        vocab.joint_potential_tensor = joint_potential_data
        vocab.tree = FragmentTree.from_list(tree_data)
        vocab.tree.smiles_and_atom_idx_to_potential = tree_smi_potential_data
        vocab.fp_tensor = fingerprint_data
        vocab.cosine_matrix = cosine_matrix_data
        vocab.tgt_cosine_matrix = tgt_cosine_matrix
        return vocab

    @staticmethod
    def load(path, message=True):
        if message:
            print(f"Loading vocabulary from {path}", end='...')
        data = dill.load(open(path, 'rb'))
        if message:
            print("done")
        vocab = Vocab.get_vocab_from_data(data)
        if message:
            print(f"Vocabulary size (>={vocab.threshold}): {len(vocab)}")
        return vocab

    def assign_vocab(self, mol):
        fragment_group = self.fragmentizer.split_molecule(mol)

        frag_to_idx = {}
        for i, fragment in enumerate(fragment_group):

            if tuple(fragment) in frag_to_idx:
                continue
            token_idx = self.idx_to_token.inv.get(fragment.smiles, -1)
            if token_idx == -1:
                ring_info = fragment.mol.GetRingInfo()
                # Nonvalid fragment if it contains a ring
                try:
                    num_rings = ring_info.NumRings()
                except Exception as e:
                    raise ValueError(e)
                    fragment.mol = Chem.MolFromSmiles(fragment.smiles)
                    ring_info = fragment.mol.GetRingInfo()
                    num_rings = ring_info.NumRings()
                if num_rings > 0:
                    raise ValueError("Error: Ring not in vocabulary.")
            frag_to_idx[fragment.smiles] = token_idx


        
        # atom_scores = calculate_advanced_scores(mol)
        # max_atom_score_tuple = max(atom_scores, key=lambda x: x[2])
        # max_atom_score_idx = max_atom_score_tuple[0]
        max_atom_score_idx = 0

        ori_frag_idx, ori_atom_idx = fragment_group.match_atom_map(max_atom_score_idx)

        vocab_list = []
        visited = set()
        current_frag_idx = -1

        def dfs(parent_info, start_bond_info):
            nonlocal current_frag_idx
            frag_idx, s_bond_idx = start_bond_info
            fragment: Fragment = fragment_group[frag_idx]
            atom_mapping = fragment.atom_map

            vocab_idx = frag_to_idx[fragment.smiles]
            if vocab_idx == -1:
                current_frag_idx += 1
                order_frag_idx = current_frag_idx
                visited.update(atom_mapping)
                if s_bond_idx == -1:
                    start_atom_idx = atom_mapping.index(max_atom_score_idx)
                else:
                    start_atom_idx = None
                result =  self.tree.search(fragment, s_bond_idx, start_atom_idx=start_atom_idx)
                if result is None:
                    return None
                root_next, sub_vocab_list, local_atom_map = result
                current_bond_pos = root_next[2][1]
                merge_bond_poses = []
                for sub_vocab_idx, sub_vocab in enumerate(sub_vocab_list):
                    tmp_frag = sub_vocab['frag']
                    if tmp_frag.smiles not in frag_to_idx:
                        frag_to_idx[tmp_frag.smiles] = self[tmp_frag.smiles]
                    sub_vocab['idx'] = frag_to_idx[tmp_frag.smiles]

                    for i, next_v in enumerate(sub_vocab['next']):
                        sub_vocab['next'][i] = (next_v[0], next_v[1], (next_v[2][0] + current_frag_idx, next_v[2][1]))

                    for frag_bond in tmp_frag.bond_list:
                        merge_bond_poses.append((sub_vocab_idx, frag_bond.id))

                    # vocab_list.append(sub_vocab)
                    vocab_list.append({'frag': tuple(sub_vocab['frag']), 'idx': sub_vocab['idx'], 'next': sub_vocab['next']})
                if root_next[2][1] != -1:
                    merge_bond_poses.remove((root_next[2][0], root_next[2][1]))
                for sub_vocab_idx, sub_vocab in enumerate(sub_vocab_list):
                        for i, next_v in enumerate(sub_vocab['next']):
                            merge_bond_poses.remove((sub_vocab_idx, next_v[0]))
                            merge_bond_poses.remove((next_v[2][0] - current_frag_idx, next_v[2][1]))
                    
                next_atom_infoes = []
                tmp = defaultdict(list)
                for i, (sub_vocab_idx, bond_pos) in enumerate(merge_bond_poses):
                    atom_idx = local_atom_map.inv[(sub_vocab_idx, sub_vocab_list[sub_vocab_idx]['frag'].bond_list[bond_pos].atom_idx)]
                    bond_token = sub_vocab_list[sub_vocab_idx]['frag'].bond_list[bond_pos].token

                    vocab_idx = sub_vocab_idx+current_frag_idx
                    tmp[(atom_idx, bond_token)].append((vocab_idx, bond_pos))
                
                for frag_bond in fragment.bond_list:
                    if frag_bond.id == s_bond_idx:
                        continue
                    vocab_idx, bond_pos = tmp[(fragment.atom_map[frag_bond.atom_idx], frag_bond.token)].pop(0)
                    next_frag_idx, next_bond_pos = fragment_group.get_neighbor(frag_idx, frag_bond.id)

                    next_atom_infoes.append(((vocab_idx, bond_pos), bond_token, (next_frag_idx, next_bond_pos)))
                
                current_frag_idx += len(sub_vocab_list) - 1

                for (vocab_idx, bond_pos), bond_token, (next_frag_idx, next_bond_pos) in next_atom_infoes:
                    next_frag_idx, next_bond_pos = dfs(parent_info=(vocab_idx, bond_pos), start_bond_info=(next_frag_idx, next_bond_pos))
                    vocab_list[vocab_idx]['next'].append((bond_pos, bond_token, (next_frag_idx, next_bond_pos)))
                    
            else:
                vocab_list.append({'frag': tuple(fragment), 'idx': vocab_idx, 'next': []})
                current_frag_idx += 1
                order_frag_idx = current_frag_idx
                current_bond_pos = s_bond_idx
                visited.update(atom_mapping)

                frag_neighbors = fragment_group.get_neighbors(frag_idx)
                frag_neighbors = {from_bond_pos: (to_frag_idx, to_bond_pos) for from_bond_pos, (to_frag_idx, to_bond_pos) in frag_neighbors.items() if from_bond_pos != s_bond_idx}

                for cur_bond_pos, (next_frag_idx, next_bond_pos) in frag_neighbors.items():
                    bond_token = fragment_group.bond_token_between(frag_idx, cur_bond_pos, next_frag_idx, next_bond_pos)
                    next_frag_idx, next_bond_pos = dfs(parent_info=(order_frag_idx, cur_bond_pos), start_bond_info=(next_frag_idx, next_bond_pos))
                    vocab_list[order_frag_idx]['next'].append((cur_bond_pos, bond_token, (next_frag_idx, next_bond_pos)))

                
            return order_frag_idx, current_bond_pos
            
        result = dfs(parent_info=(-1,-1), start_bond_info=(ori_frag_idx, -1))

        if result is None:
            return None
        
        return vocab_list
    

    def tensorize(self, mol, max_seq_len = 100):
        vocab_tree = self.assign_vocab(mol)
        # print('\n'.join([str(i) + ': ' + str(vt) for i, vt in enumerate(vocab_tree)]))
        if vocab_tree is None:
            return None
        
        vocab_tensor = torch.full((max_seq_len,), -1, dtype=torch.int64)
        order_tensor = torch.full((max_seq_len, 5), -1, dtype=torch.int64) # (parent_idx, parent_atom_pos, atom_pos, bond_type[0~2], level)
        mask_tensor =  torch.zeros(max_seq_len, dtype=torch.bool)  # 初期値は False
        mask_tensor[:len(vocab_tree)] = True

        parent_data = {}
        parent_data[0] = [-1, -1, -1, -1, -1]
        sorting_order = [0]
        num = 0
        while num < len(vocab_tree):
            parent_idx = sorting_order[num]
            parent_token_idx = vocab_tree[parent_idx]['idx']
            num += 1
            for next_vocab in sorted(vocab_tree[parent_idx]['next'], key=lambda x: x[0]):
                parent_bond_pos = next_vocab[0]
                next_idx = next_vocab[2][0]
                next_token_idx = vocab_tree[next_idx]['idx']
                bond_pos = next_vocab[2][1]
                sorting_order.append(next_idx)
                parent_atom_idx = vocab_tree[parent_idx]['frag'][parent_bond_pos*2+1]
                atom_idx = vocab_tree[next_idx]['frag'][bond_pos*2+1]
                parent_atom_pos = torch.where(self.joint_potential_tensor[parent_token_idx,:,0]==parent_atom_idx)[0]
                atom_pos = torch.where(self.joint_potential_tensor[next_token_idx,:,0]==atom_idx)[0]
                num_bond = token_to_num_bond(next_vocab[1]) - 1
                parent_data[sorting_order.index(next_idx)] = [sorting_order.index(parent_idx), parent_atom_pos, atom_pos, num_bond, -1]

        max_level = max([parent_data[i][4] for i in range(len(parent_data))])

        for i, vocab_i in enumerate(sorting_order):
            vocab = vocab_tree[vocab_i]
            vocab_tensor[i] = vocab['idx']
            order_tensor[i] = torch.tensor(parent_data[i], dtype=torch.int64)
        
        # level
        unique_parent_indices = torch.unique(order_tensor[:, 0])
        unique_parent_indices = unique_parent_indices[unique_parent_indices >= 0]
        all_parent_indices = torch.arange(torch.sum(mask_tensor))
        leave_indices = torch.tensor(np.setdiff1d(all_parent_indices.numpy(), unique_parent_indices.numpy()))

        order_tensor[leave_indices, 4] = 0
        for i in reversed(all_parent_indices):
            parent_idx = order_tensor[i, 0]
            parent_level = order_tensor[parent_idx, 4]
            level = order_tensor[i, 4] + 1
            if parent_idx >= 0:
                order_tensor[parent_idx, 4] = max(parent_level, level)

        return vocab_tensor, order_tensor, mask_tensor

    def detensorize(self, vocab_tensor, order_tensor, mask_tensor):
        # 空の vocab_tree を作成
        nodes = []

        # mask_tensor で有効なインデックスのみ処理
        valid_indices = mask_tensor.nonzero(as_tuple=True)[0]

        # vocab_tree を再構築
        for idx in valid_indices:
            token_idx = vocab_tensor[idx].item()
            parent_idx, parent_atom_pos, atom_pos, bond_type_num, level = \
                order_tensor[idx].tolist()
            if parent_atom_pos != -1:
                parent_atom_idx = self.joint_potential_tensor[nodes[parent_idx]['idx'], parent_atom_pos, 0].item()
            else:
                parent_atom_idx = -1
            
            if atom_pos != -1:
                atom_idx = self.joint_potential_tensor[token_idx, atom_pos, 0].item()
            else:
                atom_idx = -1
            
            # ノードのデータ構造
            smiles = self[token_idx]
            node = {
                'smi': smiles,
                'bonds': [],
                'idx': token_idx,
                'next': [],
            }
            # frag = Fragment.from_tuple(frag_info)
            # node = {
            #     'frag': frag, 
            #     'idx': token_idx,
            #     'next': []
            # }
            nodes.append(node)

            # 親ノードに 'next' 情報を追加
            if parent_idx >= 0:
                nodes[parent_idx]['next'].append((parent_atom_idx, num_bond_to_token(bond_type_num+1), (idx.item(), atom_idx)))
                nodes[parent_idx]['bonds'].append((parent_atom_idx, num_bond_to_token(bond_type_num+1)))
                nodes[idx.item()]['bonds'].append((atom_idx, num_bond_to_token(bond_type_num+1)))

        # vocab_treeを構築する
        vocab_tree = []
        for node in nodes:
            smi = node['smi']
            bond_list = FragBondList(node['bonds'])
            frag = Fragment(smi, bond_list)
            vocab_tree.append({'frag': frag, 'idx': node['idx'], 'next': []})

        bond_pos_frags = []
        for idx, node in enumerate(nodes):
            for atom_idx, bond_token, (child_idx, child_atom_idx) in node['next']:
                for bond_id in vocab_tree[idx]['frag'].bond_list.get_bond_ids(atom_idx, bond_token):
                    if (idx, bond_id) not in bond_pos_frags:
                        bond_pos_frags.append((idx, bond_id))
                        break
                else:
                    raise ValueError(f"Error: bond not found in next fragment.")
                
                child_frag = vocab_tree[child_idx]['frag']
                for child_bond_id in child_frag.bond_list.get_bond_ids(child_atom_idx, bond_token):
                    if (child_idx, child_bond_id) not in bond_pos_frags:
                        bond_pos_frags.append((child_idx, child_bond_id))
                        break
                else:
                    raise ValueError(f"Error: bond not found in child fragment.")
                
                vocab_tree[idx]['next'].append((bond_id, bond_token, (child_idx, child_bond_id)))
                    
        

        # 再構築された vocab_tree を返す
        mol = self.vocab_tree_to_mol(vocab_tree)
        return mol

    def vocab_tree_to_mol(self, vocab_tree):
        merge_bond_poses = []
        fragments = []
        for frag_id1, vocab in enumerate(vocab_tree):
            for next_info in vocab['next']:
                bond_pos1, bond_type, (frag_id2, bond_pos2) = next_info
                merge_bond_poses.append(((frag_id1, bond_pos1), bond_type, (frag_id2, bond_pos2)))
            fragments.append(vocab['frag'])
        
        merged_frag =  merge_fragment_info(fragments, merge_bond_poses)

        # 分子を正規化して返す
        mol = Chem.MolFromSmiles(merged_frag.smiles)
        return mol
    
    def tensorize_by_level(self, vocab_tensor, order_tensor, mask_tensor):
        """
        Create tree structures for all levels in the data.
        Args:
            vocab_tensor: Tensor containing vocabulary indices. Shape: [max_seq_len]
            order_tensor: Tensor with parent-child relationship and metadata. Shape: [max_seq_len, 5]
            mask_tensor: Tensor indicating valid entries. Shape: [max_seq_len]

        Returns:
            level_trees: A dictionary where keys are levels and values are tuples:
                        (vocab_tensor_filtered, order_tensor_filtered, mask_tensor_filtered)
        """
        # Get all unique levels from order_tensor[:, 4]
        levels = torch.arange(torch.max(order_tensor[:, 4]) + 1)

        tensor_by_branch = {}
        indices_by_level = defaultdict(list)
        child_ids = defaultdict(list)

        for i, order in enumerate(order_tensor):
            if not mask_tensor[i]:
                break
            if order[0] == -1:
                continue

            child_ids[order[0].item()].append(i)

        for i, order in enumerate(order_tensor):
            indices_by_level[order[4].item()].append(i)

        for i in indices_by_level[0]:
            level = 0
            tmp_vocab_tensor = torch.full_like(vocab_tensor, -1)
            tmp_order_tensor = torch.full_like(order_tensor, -1) # (parent_idx, parent_atom_pos, atom_pos, bond_type[0~2], level)
            tmp_mask_tensor = torch.zeros_like(mask_tensor)
            tmp_vocab_tensor[0] = vocab_tensor[i]
            tmp_order_tensor[0, 4] = level
            tmp_mask_tensor[0] = True

            tensor_by_branch[i] = {
                'tensor': [tmp_vocab_tensor, tmp_order_tensor, tmp_mask_tensor],
                'level': 0
            }

        for level in levels[1:]:
            level = level.item()
            for i in indices_by_level[level]:
                tmp_vocab_arr = []
                tmp_order_arr = []
                tmp_mask_arr = []

                next_indices = [i]
                tmp_vocab_arr.append(vocab_tensor[i].item())
                tmp_order_arr.append([-1, -1, -1, -1, level])
                tmp_mask_arr.append(True)
                origin_idx_to_new_idx = {i: 0}

                while len(next_indices) > 0:
                    next_idx = next_indices.pop(0)
                    for child_idx in child_ids[next_idx]:
                        origin_idx_to_new_idx[child_idx] = len(tmp_vocab_arr)
                        tmp_vocab_arr.append(vocab_tensor[child_idx].item())
                        tmp_order = [
                            origin_idx_to_new_idx[order_tensor[child_idx, 0].item()], # parent_idx
                            order_tensor[child_idx, 1].item(), # parent_atom_pos
                            order_tensor[child_idx, 2].item(), # atom_pos
                            order_tensor[child_idx, 3].item(), # bond_type
                            order_tensor[child_idx, 4].item()
                        ]
                        tmp_order_arr.append(tmp_order)
                        tmp_mask_arr.append(True)
                        next_indices.append(child_idx)

                _tmp_vocab_tensor = torch.tensor(tmp_vocab_arr, dtype=torch.int64)
                tmp_vocab_tensor = torch.full_like(vocab_tensor, -1)
                tmp_vocab_tensor[:len(_tmp_vocab_tensor)] = _tmp_vocab_tensor
                _tmp_order_tensor = torch.tensor(tmp_order_arr, dtype=torch.int64)
                tmp_order_tensor = torch.full_like(order_tensor, -1)
                tmp_order_tensor[:len(_tmp_order_tensor)] = _tmp_order_tensor
                _tmp_mask_tensor = torch.tensor(tmp_mask_arr, dtype=torch.bool)
                tmp_mask_tensor = torch.zeros_like(mask_tensor)
                tmp_mask_tensor[:len(_tmp_mask_tensor)] = _tmp_mask_tensor

                tensor_by_branch[i] = {
                    'tensor': [tmp_vocab_tensor, tmp_order_tensor, tmp_mask_tensor],
                    'level': level
                }

        return tensor_by_branch





    def get_graph(self, fragment: Fragment):
            # # get the graph
            # self.vocab_idx_to_graph = {}
            # for v, idx in tqdm(self.vocab.items(), desc='VocabToGraph', mininterval=0.5):
            #     if v in Vocab.TOKENS:
            #         node_tensor, edge_tensor, frag_bond_tensor = \
            #             torch.zeros(0), \
            #                 torch.zeros(0, 3, dtype=torch.int32), \
            #                     torch.zeros(0, 3, dtype=torch.int32)
            #     else:
            #         node_tensor, edge_tensor, frag_bond_tensor = vocab_tuple_to_graph[v]
                
            #     self.vocab_idx_to_graph[idx] = (node_tensor, edge_tensor, frag_bond_tensor)

        node_tensor = torch.tensor([self.symbol_to_idx[atom.GetSymbol()] for atom in fragment.mol.GetAtoms()], dtype=torch.int32)
        frag_bond_tensor = fragment.get_frag_bond_tensor() #[atom_nums, 3 (fragment bond counter['-', '=', '#'])]

        edge_list = []
        for bond in fragment.mol.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            bond_type = bond.GetBondType()
            bond_num = token_to_num_bond(chem_bond_to_token(bond_type))
            edge_list.append((atom1, atom2, bond_num))
        
        edge_tensor = torch.tensor(edge_list, dtype=torch.int32)

        return node_tensor, edge_tensor, frag_bond_tensor

    def get_counter_tensor(self, fragment: Fragment):
        atom_counter_tensor = torch.zeros(len(self.symbol_to_idx), dtype=torch.float32)
        if fragment is not None:
            for atom in fragment.mol.GetAtoms():
                atom_counter_tensor[self.symbol_to_idx[atom.GetSymbol()]] += 1.0
        
        inner_bond_counter_tensor = torch.zeros(4, dtype=torch.float32)
        if fragment is not None:
            for bond in fragment.mol.GetBonds():
                inner_bond_counter_tensor[token_to_num_bond(chem_bond_to_token(bond.GetBondType()))-1] += 1.0

        if fragment is not None:
            outer_bond_cnt_tensor = torch.tensor([len(fragment.bond_list)], dtype=torch.float32)
        else:
            outer_bond_cnt_tensor = torch.tensor([0], dtype=torch.float32)

        return atom_counter_tensor, inner_bond_counter_tensor, outer_bond_cnt_tensor
    


def calculate_advanced_scores(smiles_or_mol: str, sort=False):
    """
    Calculate advanced scores for each atom in a molecule based on:
    - Symbol score (atomic type)
    - Join type score (bond type)
    - Distance from molecule center
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    elif isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
    else:
        raise ValueError("Invalid input type. Expected SMILES string or RDKit Mol object.")
    
    if mol is None:
        raise ValueError(f"Invalid mol")

    total_atoms = mol.GetNumAtoms()
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)

    # Updated symbol_scores with Carbon having the highest score
    symbol_scores = {
        "C": 2.0, "N": 1.5, "O": 1.4, "S": 1.3, "P": 1.2, 
        "F": 1.1, "Cl": 1.0, "Br": 0.9, "I": 0.8, 
        "Si": 0.7, "B": 0.6, "Li": 0.5, "Na": 0.4, 
        "K": 0.3, "Mg": 0.2, "Ca": 0.1
    }

    bond_scores = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 1.5, Chem.BondType.TRIPLE: 2.0, Chem.BondType.AROMATIC: 2.5}

    scores = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        if symbol == "H":  # Skip hydrogen
            continue
        max_distance = distance_matrix[atom_idx].max()  # Average distance
        symbol_score = symbol_scores.get(symbol, 0.1)  # Default to minimum score for unknown atoms

        # Calculate join type score
        join_type_score = sum(
            bond_scores.get(bond.GetBondType(), 0) for bond in atom.GetBonds()
        )

        # Calculate the final score
        atom_score = sum(
            ((max_distance - dist) / max_distance / join_type_score * symbol_score) / total_atoms
            for dist in distance_matrix[atom_idx]
        )
        scores.append((atom_idx, symbol, atom_score))
    
    if sort:
        scores = sorted(scores, key=lambda x: x[2], reverse=True)
    return scores


def merge_fragment_info(fragments: list[Fragment], merge_bond_poses, atom_id_list = None):
    """
    Merge multiple molecular fragments into a single molecule by combining and bonding them.
    
    Args:
        frag_infoes (list of tuples): Each tuple contains fragment information. 
            The first element is the SMILES string of the fragment. Subsequent elements are:
            (smiles, atom_idx1, bond_type1, atom_idx2, bond_type2, ...)
                - The atom indices involved in bonds
                - The bond types (e.g., '-', '=', etc.)
        merge_bond_poses (list of tuples): Specifies the bonds to be created between fragments.
            Each tuple contains:
            ((frag_idx1, bond_pos1), bond_type, (frag_idx2, bond_pos2))
                - Position of the first fragment and its bond index
                - Bond type (e.g., '-', '=', etc.)
                - Position of the second fragment and its bond index
        atom_id_list (list of lists, optional): Maps fragment atom indices to global atom IDs.
            If not provided, default indices [0, 1, ..., N] are used for each fragment.

    Returns:
        tuple: A tuple containing:
            - final_frag_info (tuple): The SMILES string of the combined molecule and bond information.
            - final_atom_map (bidict): Maps global atom indices in the combined molecule to fragment indices and atom IDs.
    """

    # Convert SMILES to RDKit molecules for each fragment
    mols = [Chem.MolFromSmiles(frag.smiles) for frag in fragments]

    # If no atom_id_list is provided, use default atom indices for each fragment
    if atom_id_list is None:
        atom_id_list = [list(range(mol.GetNumAtoms())) for mol in mols]

    # Initialize atom mapping and remaining bond positions
    atom_map = bidict()
    remaining_bond_poses = []
    offset = 1

    # Combine molecules and assign atom map numbers
    for i, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom.SetAtomMapNum(atom_idx + offset)  # Assign unique atom map numbers
            atom_map[atom_idx + offset] = (i, atom_idx)
            
        if i == 0:
            combined_mol = copy.deepcopy(mol)  # Start with the first fragment
        else:
            combined_mol = Chem.CombineMols(combined_mol, mol)  # Add subsequent fragments

        # Track remaining bonds in the fragment
        for frag_bond in fragments[i].bond_list:
            remaining_bond_poses.append((i, frag_bond.id))
        offset += mol.GetNumAtoms()  # Update offset for the next fragment

    # Convert the combined molecule to an editable RWMol
    combined_rwmol = Chem.RWMol(combined_mol)

    # Add specified bonds between fragments
    for i, (joint_pos1, bond_token, joint_pos2) in enumerate(merge_bond_poses):
        frag_idx1, bond_pos1 = joint_pos1
        map_number1 = atom_map.inv[(frag_idx1, fragments[frag_idx1].bond_list[bond_pos1].atom_idx)]

        frag_idx2, bond_pos2 = joint_pos2
        map_number2 = atom_map.inv[(frag_idx2, fragments[frag_idx2].bond_list[bond_pos2].atom_idx)]

        # Find atom indices by map number
        atom_idx1 = next(atom.GetIdx() for atom in combined_rwmol.GetAtoms() if atom.GetAtomMapNum() == map_number1)
        atom_idx2 = next(atom.GetIdx() for atom in combined_rwmol.GetAtoms() if atom.GetAtomMapNum() == map_number2)

        atom1 = combined_rwmol.GetAtomWithIdx(atom_idx1)
        atom2 = combined_rwmol.GetAtomWithIdx(atom_idx2)
        bond_type = token_to_chem_bond(bond_token)  # Convert bond type to RDKit format
        combined_rwmol.AddBond(atom_idx1, atom_idx2, bond_type)  # Add bond
        combined_rwmol = remove_Hs(combined_rwmol, atom1, atom2, bond_type)  # Remove hydrogens
        remaining_bond_poses.remove((frag_idx1, bond_pos1))
        remaining_bond_poses.remove((frag_idx2, bond_pos2))

    # Generate the final combined molecule and SMILES
    combined_mol = combined_rwmol.GetMol()
    atom_map2 = bidict() # combined_order -> (frag_idx, pre_atom_idx)
    for i, atom in enumerate(combined_mol.GetAtoms()):
        atom_map2[i] = atom_map[atom.GetAtomMapNum()]
    for atom in combined_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(combined_mol, isomericSmiles=True)

    # Extract atom order from SMILES
    atom_order = list(map(int, combined_mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(",")))

    new_atom_maps = bidict({i: atom_map2[order] for i, order in enumerate(atom_order)})

    # Collect remaining bond information
    # (frag_idx, atom_idx) -> atom_map_num
    bond_list = []
    for frag_idx, bond_pos in remaining_bond_poses:
        frag_bond = fragments[frag_idx].bond_list[bond_pos]
        bond_list.append((atom_order.index(atom_map2.inv[(frag_idx, atom_id_list[frag_idx][frag_bond.atom_idx])]), frag_bond.token))
    bond_list = FragBondList(bond_list)

    # Create the new fragment information
    new_fragment = Fragment(smiles, bond_list, list(range(len(new_atom_maps))))

    return new_fragment