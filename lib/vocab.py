import dill
from collections import Counter, defaultdict
from bidict import bidict
from tqdm import tqdm
import torch

from .utils import *
from .calc import *
from .calc_formula import *
from .fragment_group import FragmentGroup, Fragment, FragBondList, FragmentBond
from .fragmentizer import Fragmentizer, merge_fragments
from .motif import *


class Vocab:
    BOS = '<BOS>'
    PAD = '<PAD>'
    UNK = '<UNK>'
    
    TOKENS = [BOS, PAD, UNK]

    def __init__(self, attachment_counter_file, attachment_threshold, attachment_collapse_threshold, save_path=None):
        self.attachment_threshold = attachment_threshold
        self.attachment_collapse_threshold = attachment_collapse_threshold
        max_attachment_cnt = 0
        if attachment_counter_file is not None:
            print("Counting motif", end='...')
            motif_attachment:list[Motif] = []
            if attachment_counter_file.endswith('.pkl'):
                smiles_counter = dill.load(open(attachment_counter_file, 'rb'))
                smiles_list = [smiles for smiles, cnt in smiles_counter.items() if cnt >= attachment_threshold]
            else:
                with open(attachment_counter_file, 'r') as f:
                    total_lines = sum(1 for _ in f)

                total_motifs = []
                filename, ext = os.path.splitext(attachment_counter_file)
                sub_attachment_counter_file = filename + f'_sub_thres_{attachment_threshold}_{attachment_collapse_threshold}' + ext
                if not os.path.exists(sub_attachment_counter_file):
                    with open(attachment_counter_file, 'r') as f:
                        pbar = tqdm(f, desc='CountingMotif', total=total_lines, mininterval=0.1)
                        for line in pbar:
                            s = line.strip().split('\t')
                            motif_smiles = s[0]
                            motif_freq = int(s[1])
                            atm_and_freq_list = []
                            for att, freq in zip(s[2::2], s[3::2]):
                                if int(freq) >= attachment_threshold:
                                    atm_and_freq_list.append((eval(att), int(freq)))
                            if len(atm_and_freq_list) > attachment_collapse_threshold:
                                pbar.set_postfix_str(f"Collapsing {motif_smiles} {len(atm_and_freq_list)}")
                                collapsed_data = self.collapse_attachment(motif_smiles, [(Attachment.from_tuple(a), Attachment.from_tuple(a), c) for a, c in atm_and_freq_list], attachment_collapse_threshold, pbar)
                                pbar.set_postfix_str()
                                for sub_motif_smi, att_and_freq_list in collapsed_data:
                                    if len(att_and_freq_list) == 0:
                                        continue
                                    tmp_motifs = [motif_smiles, sub_motif_smi, str(sum([f for ori_a, a, f in att_and_freq_list]))]
                                    for ori_att, att, freq in att_and_freq_list:
                                        tmp_motifs.extend([str(tuple(ori_att)), str(tuple(att)), str(freq)])
                                    total_motifs.append(tmp_motifs)
                            else:
                                if len(atm_and_freq_list) == 0:
                                    continue
                                tmp_motifs = [motif_smiles, motif_smiles, str(sum([f for a, f in atm_and_freq_list]))]
                                for att, freq in atm_and_freq_list:
                                    tmp_motifs.extend([str(tuple(att)), str(tuple(att)), str(freq)])
                                total_motifs.append(tmp_motifs)
                            max_attachment_cnt = max(max_attachment_cnt, len(atm_and_freq_list))

                    with open(sub_attachment_counter_file, 'w') as f:
                        f.write('\n'.join(['\t'.join(m) for m in total_motifs]))
            print("done")

            motif_attachment = {}
            root_motifs = set()
            motif_to_root_motif = {}
            root_motif_to_motif = defaultdict(list)
            motif_and_attachment_tuple_to_ori_motif_str = {}
            motif_str_to_freq = {}
            motif_str_and_attachment_tuple_to_freq = {}
            max_attach_atom_cnt = 0
            total_lines = sum(1 for _ in open(sub_attachment_counter_file, 'r'))
            with open(sub_attachment_counter_file, 'r') as f:
                for line in tqdm(f, total=total_lines, desc='ReadingMotif', mininterval=0.1):
                    s = line.strip().split('\t')
                    root_motif_smiles = s[0]
                    motif_smiles = s[1]
                    sub_motif_freq = int(s[2])
                    motif_attachment[motif_smiles] = []
                    motif_mol = Chem.MolFromSmiles(motif_smiles)
                    is_ring = (motif_mol.GetRingInfo().NumRings() > 0)
                    if is_ring:
                        motif_str_to_freq[motif_smiles] = sub_motif_freq
                    else:
                        motif_str_to_freq[motif_smiles] = -sub_motif_freq
                    for ori_att, att, freq in zip(s[3::3], s[4::3], s[5::3]):
                        ori_att_tuple = eval(ori_att)
                        att_tuple = eval(att)
                        for a in att_tuple:
                            if a[0] != '':
                                max_attach_atom_cnt = max(max_attach_atom_cnt, Chem.MolFromSmiles(a[0]).GetNumAtoms())
                        ori_attachment = Attachment.from_tuple(ori_att_tuple)
                        attachment = Attachment.from_tuple(att_tuple)
                        ori_motif = Motif(root_motif_smiles, ori_attachment)
                        motif = Motif(motif_smiles, attachment)
                        motif_attachment[motif_smiles].append(motif.attachment)
                        motif_and_attachment_tuple_to_ori_motif_str[(motif_smiles, motif.attachment.to_tuple())] = str(ori_motif)
                        motif_str_and_attachment_tuple_to_freq[(motif_smiles, motif.attachment.to_tuple())] = int(freq)


                    root_motifs.add(motif_smiles)
                    motif_to_root_motif[motif_smiles] = root_motif_smiles
                    root_motif_to_motif[root_motif_smiles].append(motif_smiles)

            root_motifs = list(root_motifs)
            self.root_motif_to_motif = dict(root_motif_to_motif)

            # generate vocabulary
            sorted_motif_str = sorted(
                motif_attachment.keys(),
                key=lambda x: (
                    0 if motif_str_to_freq[x] > 0 else 1,  # 正の値を優先（0: 正の値, 1: 負の値）
                    abs(motif_str_to_freq[x]) if motif_str_to_freq[x] > 0 else -motif_str_to_freq[x]  # 正の値は昇順、負の値は降順
                )
            )
            idx_to_motif_token = bidict({i:v for i, v in enumerate(Vocab.TOKENS)})
            idx_to_motif_token.update({i+len(Vocab.TOKENS):smi for i, smi in enumerate(sorted_motif_str)})
            self.idx_to_motif_token = idx_to_motif_token
            self._set_token_idx()

            # generate attachment tensor
            idx_to_attachment_tuple = bidict()
            ori_motif_str_to_motif_and_attachment_tuple = bidict()
            for i, motif_token in tqdm(self.idx_to_motif_token.items(), desc='AttachmentFromVocab'):
                if motif_token in Vocab.TOKENS:
                    continue
                sorted_attachment = sorted(
                    motif_attachment[motif_token],
                    key=lambda att: motif_str_and_attachment_tuple_to_freq[(motif_token, att.to_tuple())]
                )
                for j, att in enumerate(sorted_attachment):
                    idx_to_attachment_tuple[(i,j)] = (motif_token, att.to_tuple())
                    ori_motif_str = motif_and_attachment_tuple_to_ori_motif_str[(motif_token, att.to_tuple())]
                    ori_motif_str_to_motif_and_attachment_tuple[ori_motif_str] = (i, j)
                    freq = motif_str_and_attachment_tuple_to_freq[(motif_token, att.to_tuple())]
            self.idx_to_attachment_tuple = idx_to_attachment_tuple
            self.ori_motif_str_to_motif_and_attachment_tuple = ori_motif_str_to_motif_and_attachment_tuple

            # attachment cnt tensor
            attachment_cnt_tensor = []
            for idx, token in tqdm(self.idx_to_motif_token.items(), desc='AttachmentFromVocab'):
                if token in Vocab.TOKENS:
                    attachment_cnt_tensor.append(0)
                else:
                    attachment_cnt = len(motif_attachment[token])
                    attachment_cnt_tensor.append(attachment_cnt)
            self.attachment_cnt_tensor = torch.tensor(attachment_cnt_tensor, dtype=torch.int64)
            
            self.fragmentizer = Fragmentizer(max_attach_atom_cnt)
            self.max_attach_atom_cnt = max_attach_atom_cnt

        if save_path:
            print(f"Saving vocabulary to {save_path}", end='...')
            self.save(save_path)
            print("done")
            print(f"Vocabulary size (>={self.attachment_threshold}): {len(self)}, MaxAttachmentSize: {attachment_collapse_threshold}, MaxAttachAtomCnt: {max_attach_atom_cnt}")

    def __len__(self):
        return len(self.idx_to_motif_token)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.idx_to_motif_token[key]
        elif isinstance(key, str):
            return self.idx_to_motif_token.inv[key]
        elif isinstance(key, Motif):
            if str(key) not in self.ori_motif_str_to_motif_and_attachment_tuple:
                return ('', -1), ((), -1)
            motif_idx, att_idx = self.ori_motif_str_to_motif_and_attachment_tuple[str(key)]
            motif_token = self.idx_to_motif_token[motif_idx]
            att_tuple = self.idx_to_attachment_tuple[(motif_idx, att_idx)]
            return (motif_token, motif_idx), (att_tuple, att_idx)
        elif isinstance(key, tuple) and len(key) == 2 and all(isinstance(i, int) for i in key):
            motif_smiles, attachment_tuple = self.idx_to_attachment_tuple[key]
            return Motif(motif_smiles, Attachment.from_tuple(attachment_tuple))
        else:
            raise ValueError(f"Invalid key type: {type(key)}")
    
    def _set_token_idx(self):
        self.bos = self[self.BOS]
        self.pad = self[self.PAD]
        self.unk = self[self.UNK]

    def get_data_to_save(self):
        data_to_save = {
            'token': self.idx_to_motif_token,
            'idx_to_attachment_tuple': self.idx_to_attachment_tuple,
            'ori_motif_str_to_motif_and_attachment_tuple': self.ori_motif_str_to_motif_and_attachment_tuple,
            'root_motif_to_motif': self.root_motif_to_motif,
            'attachment_cnt': self.attachment_cnt_tensor,
            'max_attach_atom_cnt': self.max_attach_atom_cnt,
            'attachment_threshold': self.attachment_threshold,
            'attachment_collapse_threshold': self.attachment_collapse_threshold,
            'bos': self.bos,
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
        idx_to_attachment_tuple_data = data['idx_to_attachment_tuple']
        ori_motif_str_to_motif_and_attachment_tuple_data = data['ori_motif_str_to_motif_and_attachment_tuple']
        root_motif_to_motif_data = data['root_motif_to_motif']
        attachment_cnt_data = data['attachment_cnt']
        max_attach_atom_cnt = data['max_attach_atom_cnt']
        attachment_threshold = data['attachment_threshold']
        attachment_collapse_threshold = data['attachment_collapse_threshold']
        vocab = Vocab(None, attachment_threshold, attachment_collapse_threshold)
        vocab.idx_to_motif_token = token_data
        vocab._set_token_idx()
        vocab.idx_to_attachment_tuple = idx_to_attachment_tuple_data
        vocab.ori_motif_str_to_motif_and_attachment_tuple = ori_motif_str_to_motif_and_attachment_tuple_data
        vocab.root_motif_to_motif = root_motif_to_motif_data
        vocab.attachment_cnt_tensor = attachment_cnt_data
        vocab.max_attach_atom_cnt = max_attach_atom_cnt
        vocab.fragmentizer = Fragmentizer(max_attach_atom_cnt)
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
            print(f"Vocabulary size (>={vocab.attachment_threshold}): {len(vocab)}, MaxAttachmentSize: {vocab.attachment_collapse_threshold}, MaxAttachAtomCnt: {vocab.max_attach_atom_cnt}")
        return vocab

    def convert_motif_token(self, motif:Motif):
        if motif.smiles not in self.root_motif_to_motif:
            return None

    def collapse_attachment(self, motif_smiles, attachments_and_freq:list[Attachment], attachment_collapse_threshold:int, pbar=None):
        if len(attachments_and_freq) <= attachment_collapse_threshold:
            return [[motif_smiles, attachments_and_freq]]
        
        next_datas = [[motif_smiles, attachments_and_freq]]
        completed_datas = []
        
        while len(next_datas) > 0:
            motif_smi, attes_and_freq = next_datas.pop(0)
            frag_dict = {}
            att_var_counter = Counter()
            merge_smiles_to_att_pos = {}
            for i, (ori_att, att, freq) in enumerate(attes_and_freq):
                checked_smiles = []
                for j, part in enumerate(att):
                    if part.smiles == '':
                        continue
                    if (motif_smi, part.motif_atom_idx, part.bond_token) not in frag_dict:
                        motif_frag = Fragment(motif_smi, [[part.motif_atom_idx, part.bond_token]])
                        frag_dict[(motif_smi, part.motif_atom_idx, part.bond_token)] = motif_frag
                    else:
                        motif_frag = frag_dict[(motif_smi, part.motif_atom_idx, part.bond_token)]

                    if (part.smiles, part.att_atom_idx, part.bond_token) not in frag_dict:
                        att_frag = Fragment(part.smiles, [[part.att_atom_idx, part.bond_token]])
                        frag_dict[(part.smiles, part.att_atom_idx, part.bond_token)] = att_frag
                    else:
                        att_frag = frag_dict[(part.smiles, part.att_atom_idx, part.bond_token)]

                    merge_frag, _ = merge_fragments([motif_frag, att_frag], [((0, 0), part.bond_token, (1, 0))])
                    if merge_frag.smiles in checked_smiles:
                        continue
                    checked_smiles.append(merge_frag.smiles)
                    att_var_counter[merge_frag.smiles] += 1
                    if merge_frag.smiles not in merge_smiles_to_att_pos:
                        merge_smiles_to_att_pos[merge_frag.smiles] = {}
                    merge_smiles_to_att_pos[merge_frag.smiles][i] = j
            
            if len(att_var_counter) == 0:
                continue
            top_smiles = att_var_counter.most_common(1)[0][0]
            new_datas = defaultdict(list)
            for i, (ori_att, att, freq) in enumerate(attes_and_freq):
                if i in merge_smiles_to_att_pos[top_smiles]:
                    motif = Motif(motif_smi, att)
                    merged_att_idx = merge_smiles_to_att_pos[top_smiles][i]
                    new_motif = motif.merge_attachment([merged_att_idx])
                    new_datas[new_motif.smiles].append((ori_att, new_motif.attachment, freq))
                else:
                    new_datas[motif_smi].append((ori_att, att, freq))
            
            for smi, att_freq in new_datas.items():
                if len(att_freq) > attachment_collapse_threshold:
                    next_datas.append([smi, att_freq])
                else:
                    completed_datas.append([smi, att_freq])
                    if pbar:
                        comp_cnt = sum([len(cd[1]) for cd in completed_datas])
                        pbar.set_postfix_str(f"Collapsing {motif_smiles} ({comp_cnt/len(attachments_and_freq):.2%})%")
        
        return completed_datas


    def assign_vocab(self, mol):
        motifs, fragment_group = self.fragmentizer.split_to_motif(mol)

        error_message = ''
        motif_att_idx_pair = []
        for i, motif in enumerate(motifs):
            (motif_token, motif_idx), (att_tuple, att_idx) = self[motif]
            if motif_idx == -1 or att_idx == -1:
                error_message += f", '{motif}' not in vocabulary"
            else:
                res_motif_smiles = self[motif_idx]
                res_motif = self[(motif_idx, att_idx)]
            motif_att_idx_pair.append((motif_idx, att_idx))
            

        if len(error_message) > 0:
            error_message = f'Error: {Chem.MolToSmiles(mol, canonical=True)}' + error_message
            return 0
            raise ValueError(error_message)
        
        primary_index = sorted(
            enumerate(motif_att_idx_pair),
            key=lambda x: (x[1][0], x[1][1]) 
        )[0][0]
        return 1



        res = self[fragment_group[0]]
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
        
        merged_frag, _ =  merge_fragments(fragments, merge_bond_poses)

        # 分子を正規化して返す
        mol = Chem.MolFromSmiles(merged_frag.smiles)
        return mol
    




    def get_formula_tensor(self, mols):
        symbol_map = {}
        symbol_to_idx = {}
        cnt = 0
        for mol in tqdm(mols, desc='MolToFormula'):
            cnt += 1
            formula = get_formula(mol)
            formula_dict = formula_to_dict(formula)
            for symbol in symbol_map.keys():
                symbol_map[symbol].append(0.0)

            for symbol in formula_dict.keys():
                if symbol not in symbol_map:
                    symbol_map[symbol] = [0.0] * cnt
                    symbol_to_idx[symbol] = len(symbol_to_idx)
                symbol_map[symbol][-1] = float(formula_dict[symbol])

        formula_tensor = torch.zeros((cnt, len(symbol_to_idx)), dtype=torch.float32)
        for symbol, values in symbol_map.items():
            idx = symbol_to_idx[symbol]
            formula_tensor[:, idx] = torch.tensor(values, dtype=torch.float32)

        return formula_tensor, symbol_to_idx



        

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
    

def motif_stats(attachment_counter_file):
    with open(attachment_counter_file, 'r') as f:
        for line in tqdm(f, desc='CountingMotif', mininterval=0.5):
            s = line.strip().split('\t')
            motif_smiles = s[0]
            motif_freq = int(s[1])
            atm_list = []
            for atm, freq in zip(s[2::2], s[3::2]):
                if int(freq) >= attachment_threshold:
                    atm_list.append(eval(atm))


if __name__ == '__main__':
    pass