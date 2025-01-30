import dill
from collections import Counter, defaultdict
from bidict import bidict
from tqdm import tqdm
import torch
from torch_geometric.data import Data

from .utils import *
from .calc import *
from .calc_formula import *
from .fragment_group import FragmentGroup, Fragment, FragBondList, FragmentBond
from .fragmentizer import Fragmentizer, merge_fragments
from .motif import *
from .atom_layer import *


class Vocab:
    BOS = '<BOS>'
    PAD = '<PAD>'
    UNK = '<UNK>'
    
    TOKENS = [BOS, PAD, UNK]

    def __init__(self, attachment_counter_file, attachment_threshold, attachment_collapse_threshold, save_path=None):
        self.attachment_threshold = attachment_threshold
        self.att_size = attachment_collapse_threshold
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
            motif_str_and_attachment_tuple_to_bonding_cnt = {}
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
                        bonding_cnt = [p.smiles for p in motif.attachment.parts].count('')
                        motif_str_and_attachment_tuple_to_bonding_cnt[(motif_smiles, motif.attachment.to_tuple())] = bonding_cnt

                    root_motifs.add(motif_smiles)
                    motif_to_root_motif[motif_smiles] = root_motif_smiles
                    root_motif_to_motif[root_motif_smiles].append(motif_smiles)

            root_motifs = list(root_motifs)
            root_motif_to_motif = dict(root_motif_to_motif)

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

            # generate attachment tensor
            idx_to_attachment_tuple = bidict()
            ori_motif_str_to_motif_and_attachment_tuple = bidict()
            bonding_cnt_tensor = []
            attached_motif_index_map = torch.full((len(idx_to_motif_token), self.att_size), -1, dtype=torch.int64)
            attached_motif_count = 0
            for i, motif_token in tqdm(idx_to_motif_token.items(), desc='AttachmentFromVocab'):
                if motif_token in Vocab.TOKENS:
                    bonding_cnt_tensor.append(0)
                    attached_motif_index_map[i,0] = attached_motif_count
                    attached_motif_count += 1
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
                    bonding_cnt = motif_str_and_attachment_tuple_to_bonding_cnt[(motif_token, att.to_tuple())]
                    bonding_cnt_tensor.append(bonding_cnt)
                    attached_motif_index_map[i,j] = attached_motif_count
                    attached_motif_count += 1
            bonding_cnt_tensor = torch.tensor(bonding_cnt_tensor, dtype=torch.int64)

            # attachment cnt tensor
            attachment_cnt_tensor = [0] * len(idx_to_motif_token)
            for idx, token in tqdm(idx_to_motif_token.items(), desc='AttachmentFromVocab'):
                if token in Vocab.TOKENS:
                    attachment_cnt_tensor[idx] = 0
                else:
                    attachment_cnt = len(motif_attachment[token])
                    attachment_cnt_tensor[idx] = attachment_cnt
            attachment_cnt_tensor = torch.tensor(attachment_cnt_tensor, dtype=torch.int64)

            # generage atom layer
            atom_layer_list = [0] * attached_motif_count
            idx_to_attached_motif_fragment_tuple = bidict()
            for (i,j), (motif_token, att_tuple) in tqdm(idx_to_attachment_tuple.items(), desc='AtomLayerFromVocab'):
                motif = Motif(motif_token, Attachment.from_tuple(att_tuple))
                motif_frag = motif.to_fragment()
                nodes_with_con, edges, attr = self.fragment_to_atom_layer(motif_frag, torch.max(bonding_cnt_tensor).item())
                data = Data(x=nodes_with_con, edge_index=edges, edge_attr=attr)
                atom_layer_list[attached_motif_index_map[i,j].item()] = data
                idx_to_attached_motif_fragment_tuple[(i,j)] = motif_frag.to_tuple()



            
            self.root_motif_to_motif = root_motif_to_motif
            self.idx_to_motif_token = idx_to_motif_token
            self._set_token_idx()
            self.idx_to_attachment_tuple = idx_to_attachment_tuple
            self.ori_motif_str_to_motif_and_attachment_tuple = ori_motif_str_to_motif_and_attachment_tuple
            self.bonding_cnt_tensor = torch.tensor(bonding_cnt_tensor, dtype=torch.int64)
            self.attached_motif_index_map = attached_motif_index_map
            self.attachment_cnt_tensor = torch.tensor(attachment_cnt_tensor, dtype=torch.int64)

            self.atom_layer_list = atom_layer_list
            self.idx_to_attached_motif_fragment_tuple = idx_to_attached_motif_fragment_tuple

            self.fragmentizer = Fragmentizer(max_attach_atom_cnt)
            self.max_attach_atom_cnt = max_attach_atom_cnt

            self.ms_filter_config = None

        if save_path:
            print(f"Saving vocabulary to {save_path}", end='...')
            self.save(save_path)
            print("done")
            print(self.get_property_message())

    def __len__(self):
        return len(self.idx_to_motif_token)
    
    @property
    def shape(self):
        return (len(self), self.att_size)
    
    @property
    def attached_motif_len(self):
        return len(self.bonding_cnt_tensor)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.idx_to_motif_token[key]
        elif isinstance(key, str):
            return self.idx_to_motif_token.inv[key]
        elif isinstance(key, Motif):
            if str(key) not in self.ori_motif_str_to_motif_and_attachment_tuple:
                return -1, -1
            motif_idx, att_idx = self.ori_motif_str_to_motif_and_attachment_tuple[str(key)]
            # motif_token = self.idx_to_motif_token[motif_idx]
            # att_tuple = self.idx_to_attachment_tuple[(motif_idx, att_idx)]
            return (motif_idx, att_idx)
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
        data_to_save = {key: value for key, value in self.__dict__.items()}
        data_to_save['bos'] = self.bos
        data_to_save['pad'] = self.pad
        data_to_save['unk'] = self.unk

        return data_to_save

    def save(self, path):
        data_to_save = self.get_data_to_save()
        dill.dump(data_to_save, open(path, 'wb'))

    @staticmethod
    def get_vocab_from_data(data):
        vocab = Vocab(None, data['attachment_threshold'], data['att_size'])

        for key, value in data.items():
            setattr(vocab, key, value)

        vocab._set_token_idx()
        # vocab.fragmentizer = Fragmentizer(vocab.max_attach_atom_cnt)
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
            print(vocab.get_property_message())
        return vocab
    
    def get_property_message(self):
        return f"Vocabulary size (>={self.attachment_threshold}): {self.attached_motif_len}/{self.shape}, MaxBondingCnt: {torch.max(self.bonding_cnt_tensor).item()}, MaxAttachAtomCnt: {self.max_attach_atom_cnt}"

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

    def fragment_to_atom_layer(self, fragment: Fragment, max_attachment_holdings):
        nodes, edges, attr = atom_bond_properties_to_tensor(fragment.mol)

        connect_tensor = torch.full((len(nodes), max_attachment_holdings), -1.0, dtype=torch.float32)
        connect_tensor[:, :len(fragment.bond_list)] = 0.0
        for frag_bond in fragment.bond_list:
            connect_tensor[frag_bond.atom_idx, frag_bond.id] = 1.0

        nodes_with_con = torch.cat([nodes, connect_tensor], dim=1)

        return nodes_with_con, edges, attr


    def tensorize(self, mol, max_seq_len):
        motifs, fragment_group = self.fragmentizer.split_to_motif(mol)

        error_motif_messages = []
        motif_att_idx_pair = []
        for i, motif in enumerate(motifs):
            motif_idx, att_idx = self[motif]
            if motif_idx == -1 or att_idx == -1:
                error_motif_messages.append(f", '{motif}'")
            else:
                # res_motif_smiles = self[motif_idx]
                # res_motif = self[(motif_idx, att_idx)]
                pass
            motif_att_idx_pair.append((motif_idx, att_idx))
            

        if len(error_motif_messages) > 0:
            error_message = f'NotFoundInVocabError: {Chem.MolToSmiles(mol, canonical=True)}' + ' -> ' + ', '.join(error_motif_messages) + ' not found in vocabulary.'
            raise ValueError(error_message)
        
        primary_index = sorted(
            enumerate(motif_att_idx_pair),
            key=lambda x: (x[1][0], x[1][1]) 
        )[0][0]


        token_list = [] # (motif_id, attachment_id)
        order_list = [] # (parent_idx, parent_bond_pos, bond_pos)

        start_fragment = fragment_group[primary_index]
        next_fragments = [(-1, -1, -1, start_fragment)] # (parent_idx, parent_bond_pos, bond_pos, fragment)
        visited = set()
        while len(next_fragments) > 0:
            parent_idx, parent_bond_pos, bond_pos, frag = next_fragments.pop(0)
            current_id = len(token_list)
            visited.add(frag.id)
            token_list.append(motif_att_idx_pair[frag.id])
            order_list.append((parent_idx, parent_bond_pos, bond_pos))
            if frag is None:
                raise ValueError('Error: Ring not in vocabulary.')
            for s_bond_pos, (e_frag_idx, e_bond_pos) in sorted(fragment_group.get_neighbors(frag.id).items(), key=lambda x: x[0]):
                if e_frag_idx in visited:
                    continue
                next_fragments.append((current_id,s_bond_pos,e_bond_pos, fragment_group[e_frag_idx]))
        
        if len(token_list) > max_seq_len:
            raise ValueError(f"SeqLenError: Sequence length '{len(token_list)}' is too long: {Chem.MolToSmiles(mol, canonical=True)}")
        
        mask_list = [False] * len(token_list)
        mask_list.extend([True] * (max_seq_len - len(token_list)))
        token_list.extend([(self.pad, 0)] * (max_seq_len - len(token_list)))
        order_list.extend([(-1, -1, -1)] * (max_seq_len - len(order_list)))

        token_tensor = torch.tensor(token_list, dtype=torch.int64)
        order_tensor = torch.tensor(order_list, dtype=torch.int64)
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
        
        return token_tensor, order_tensor, mask_tensor

    def detensorize(self, token_tensor, order_tensor, mask_tensor):
        tokens = token_tensor[~mask_tensor]
        orders = order_tensor[~mask_tensor]

        fragment_list = []
        merge_bond_poses = []
        for i, (motif_idx, att_idx) in enumerate(tokens):
            motif_frag_tuple = self.idx_to_attached_motif_fragment_tuple[(motif_idx.item(), att_idx.item())]
            order = orders[i]
            
            frag = Fragment.from_tuple(motif_frag_tuple)
            fragment_list.append(frag)
            if order[0] == -1:
                continue
            merge_bond_poses.append(((order[0].item(), order[1].item()), '-', (i, order[2].item())))
        
        merged_frag, _ =  merge_fragments(fragment_list, merge_bond_poses)

        mol = Chem.MolFromSmiles(merged_frag.smiles)
        return mol


    def set_ms_filter_config(self, filter_config):
        self.ms_filter_config = {}
        self.ms_filter_config['max_exact_mass'] = filter_config['max_exact_mass']
        self.ms_filter_config['max_atom_counts'] = filter_config['max_atom_counts']
        self.ms_filter_config['adduct_types'] = filter_config['adduct_types']
        self.ms_filter_config['ppm'] = filter_config['ppm']
        self.ms_filter_config['sn_threshold'] = filter_config['sn_threshold']
        self.ms_filter_config['min_peak_number'] = filter_config['min_peak_number']
        self.ms_filter_config['filter_precursor'] = filter_config['filter_precursor']


    def tensorize_msspectra(
        self,
        spectra_df: pd.DataFrame, 
        max_seq_len:int,
        output_file:str,
        ) -> torch.Tensor:
        atoms = [e for e,cnt in self.ms_filter_config['max_atom_counts'].items() if cnt > 0]
        if 'H' not in atoms:
            atoms.append('H')
        adduct_types = list(set(self.ms_filter_config['adduct_types']))

        feature_name_to_idx = {}
        current_index = 0
        feature_name_to_idx['intensity'] = current_index
        current_index += 1
        for atom in atoms:
            feature_name_to_idx[f'PI({atom})'] = current_index
            feature_name_to_idx[f'NL({atom})'] = current_index+1
            current_index += 2
        feature_name_to_idx['unsaturation'] = current_index
        current_index += 1
        feature_name_to_idx['adduct_type'] = current_index
        current_index += 1
        feature_name_to_idx['collision'] = current_index
        current_index += 1
        empty_tensor = [0]*current_index

        total_peak = []
        total_mask = []
        indices = []
        for i, row in tqdm(spectra_df.iterrows(), total=len(spectra_df), desc='Tensorizing MS/MS spectra'):
            seq_tensor = []
            # Precursor
            formula = row['Formula']
            precursor_elements = formula_to_dict(formula)
            # tmp_tensor[feature_name_to_idx['intensity']] = 1.1
            skip = False
            # for el, cnt in precursor_elements.items():
            #     if el not in atoms:
            #         skip = True
            #         break
            #     tmp_tensor[feature_name_to_idx[f'PI({el})']] = cnt
            #     tmp_tensor[feature_name_to_idx[f'NL({el})']] = 0
            
            # if skip:
            #     continue

            # unsaturation = calc_unsaturations(precursor_elements)
            # tmp_tensor[feature_name_to_idx['unsaturation']] = unsaturation

            adduct_type = row['PrecursorType']
            if adduct_type in adduct_types:
                adduct_type_idx = adduct_types.index(adduct_type)
            else:
                continue
                

            collision_energy = -1
            try:
                if row['CollisionEnergy1'] == row['CollisionEnergy2']:
                    collision_energy = int(row['CollisionEnergy1'])
            except:
                pass

            # Fragment
            for peak_str in row['FormulaPeaks'].split(';'):
                mass, intensity, formula, unsaturaiton, ppm = peak_str.split(',')
                mass, intensity, unsaturation, ppm = float(mass), float(intensity), int(unsaturaiton), float(ppm)
                
                tmp_tensor = empty_tensor.copy()
                elements = formula_to_dict(formula)
                tmp_tensor[feature_name_to_idx['intensity']] = intensity
                for el, cnt in precursor_elements.items():
                    tmp_tensor[feature_name_to_idx[f'PI({el})']] = elements.get(el, 0)
                    tmp_tensor[feature_name_to_idx[f'NL({el})']] = cnt - elements.get(el, 0)

                tmp_tensor[feature_name_to_idx['unsaturation']] = unsaturation
                tmp_tensor[feature_name_to_idx['adduct_type']] = adduct_type_idx
                tmp_tensor[feature_name_to_idx['collision']] = collision_energy

                seq_tensor.append(tmp_tensor)

            if len(seq_tensor) < max_seq_len:
                total_mask.append([False]*len(seq_tensor) + [True]*(max_seq_len - len(seq_tensor)))
                seq_tensor += [empty_tensor]*(max_seq_len - len(seq_tensor))
            else:
                total_mask.append([False]*max_seq_len)
                indexed_seq_tensor = [(i, x) for i, x in enumerate(seq_tensor)]
                intensity_idx = feature_name_to_idx['intensity']
                indexed_seq_tensor = sorted(indexed_seq_tensor, key=lambda x: x[1][intensity_idx], reverse=True)
                indexed_seq_tensor = indexed_seq_tensor[:max_seq_len]
                indexed_seq_tensor = sorted(indexed_seq_tensor, key=lambda x: x[0])
                seq_tensor = [x[1] for x in indexed_seq_tensor]
            total_peak.append(seq_tensor)
            indices.append(row['IdxOri'])
        
        peak_tensor = torch.tensor(total_peak, dtype=torch.float32)
        indices_tensor = torch.tensor(indices, dtype=torch.int32)
        mask_tensor = torch.tensor(total_mask, dtype=torch.bool)
        
        torch.save({
            'type': 'msms_formula',
            'indices': indices_tensor,
            'peak': peak_tensor,
            'mask': mask_tensor,
            'map' : feature_name_to_idx,
            'max_seq_len': max_seq_len,
        }, output_file)
        print("Saved to", output_file)
        print("Total peaks: ", f"{len(total_peak)}/{len(spectra_df)}({len(total_peak)/len(spectra_df):.2%})")






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