from .fragment_group import *
from .fragment_edit import split_fragment, merge_fragments
from typing import Iterator

    
class AttPart:
    def __init__(self, smiles, motif_atom_idx, attachment_atom_idx, bond_token):
        self.smiles = smiles
        self.motif_atom_idx = motif_atom_idx
        self.att_atom_idx = attachment_atom_idx
        self.bond_token = bond_token
    
    def __str__(self):
        return str(self.to_tuple())
    
    def __repr__(self):
        return self.__str__()
    
    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()
    
    def to_tuple(self):
        return (self.smiles, self.motif_atom_idx, self.att_atom_idx, self.bond_token)
    

    @staticmethod
    def from_tuple(t):
        return AttPart(t[0], t[1], t[2], t[3])
    
    def to_fragment(self):
        return Fragment(self.smiles, [(self.att_atom_idx, self.bond_token)])


class Attachment:
    def __init__(self, parts:list[AttPart]):
        if parts is None:
            self.parts = None
        else:
            self.parts = sorted(parts)
    
    def __len__(self):
        return len(self.parts)
    
    def __getitem__(self, idx):
        return self.parts[idx]
    
    def __iter__(self) -> Iterator[AttPart]:
        return iter(self.parts)

    def __str__(self):
        return str(tuple(self.parts))
    
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def from_tuple(attachment:tuple[str,int,int,str]):
        arr = [AttPart.from_tuple(att) for att in sorted(attachment)]
        attachment = Attachment.from_fragment_group(None, None)
        attachment.parts = arr
        return attachment
    
    def to_tuple(self):
        return tuple([part.to_tuple() for part in self.parts])
    
    @staticmethod
    def from_fragment_group(fragment_group:FragmentGroup, motif_idx):
        if fragment_group is None:
            return Attachment(None)
        attachment_list = []
        frag = fragment_group[motif_idx]
        for frag_bond in frag.bond_list:
            neighbor_frag_id, neighbor_bond_pos = fragment_group.get_neighbor(frag.id, frag_bond.id)
            if neighbor_frag_id is None:
                attachment_list.append(('', frag_bond.atom_idx, -1, frag_bond.token))
            else:
                neighbor_atom_idx = fragment_group[neighbor_frag_id].bond_list[neighbor_bond_pos].atom_idx
                smi = fragment_group[neighbor_frag_id].smiles
                attachment_list.append((smi, frag_bond.atom_idx, neighbor_atom_idx, frag_bond.token))
        parts = [AttPart.from_tuple(att) for att in sorted(attachment_list)]
        attachment = Attachment(parts)
        return attachment
    

class Motif:
    def __init__(self, smiles, attachment):
        self.smiles = smiles
        self.attachment = attachment
        self._attached_motif = None
    
    def __str__(self):
        return f'{self.smiles}|{self.attachment}'

    def __repr__(self):
        return self.__str__()
    
    def merge_attachment(self, attachment_indices):
        new_attachment_list = []
        merged_att_idx_list = []
        new_motif_frag_bond_list = []
        for i, part in enumerate(self.attachment):
            if i in attachment_indices and part.smiles != '':
                new_motif_frag_bond_list.append((part.motif_atom_idx, part.bond_token))
                merged_att_idx_list.append(i)
            else:
                new_attachment_list.append((part.smiles, part.motif_atom_idx, part.att_atom_idx, part.bond_token))
        
        new_motif_frag = Fragment(self.smiles, new_motif_frag_bond_list)
        new_motif_frag_bond_list = [(new_motif_frag.atom_map.index(i), bond_token) for i, bond_token in new_motif_frag_bond_list]
        new_bond_poses = new_motif_frag.get_bond_poses_batch(new_motif_frag_bond_list)
        merged_infoes = []
        merge_fragment_list = [new_motif_frag]
        for i, (atom_idx, bond_token) in enumerate(new_motif_frag_bond_list):
            part = self.attachment[merged_att_idx_list[i]]
            merge_fragment_list.append(part.to_fragment())
            merged_infoes.append(((0, new_bond_poses[i]), part.bond_token, (len(merge_fragment_list)-1, 0)))
            
        merged_fragment, new_atom_map = merge_fragments(merge_fragment_list, merged_infoes)
        new_attachment2 = []
        for new_attachment in new_attachment_list:
            smi = new_attachment[0]
            motif_atom_i = new_atom_map.inv[(0, new_motif_frag.atom_map.index(new_attachment[1]))]
            att_atom_i = new_attachment[2]
            bond_token = new_attachment[3]
            new_attachment2.append(AttPart(smi, motif_atom_i, att_atom_i, bond_token))
        new_motif = Motif(merged_fragment.smiles, Attachment(new_attachment2))

        return new_motif


    def merge_all_attachment(self):
        return self.merge_attachment(list(range(len(self.attachment))))
    
    @property
    def attached_motif(self):
        if self._attached_motif is None:
            self._attached_motif = self.merge_all_attachment()
        return self._attached_motif
    
    def to_fragment(self):
        new_motif = self.attached_motif
        frag_bond_list = [(part.motif_atom_idx, part.bond_token) for part in new_motif.attachment]
        return Fragment(new_motif.smiles, frag_bond_list)
    
    @staticmethod
    def from_fragment(fragment: list[Fragment], seps:list[int,int], motif_atom_map):
        fragment_group = split_fragment(fragment, seps)
        for i, frag in enumerate(fragment_group):
            if set(frag.atom_map) == set(motif_atom_map):
                motif_idx = i
                break
        motif = Motif(fragment_group[motif_idx].smiles, Attachment.from_fragment_group(fragment_group, motif_idx))
        return motif

    @staticmethod
    def load(smiles, attachment_str):
        motif = Motif(None, None, None)
        motif.smiles = smiles
        motif.attachment = Attachment.load(attachment_str)
        return motif