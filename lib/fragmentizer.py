from rdkit import Chem
from collections import defaultdict, Counter
import copy
from typing import List, Tuple

from .fragment_edit import *
from .utils import *
from .motif import Motif


# function group class
class FunctionalGroup:
    def __init__(self, smarts):
        self.smarts = smarts
        self.mol = Chem.MolFromSmarts(smarts)
        self.wildcard_indices = self.find_wildcard_positions(self.mol)
        self.num_atoms = self.mol.GetNumAtoms() - len(self.wildcard_indices)
    
    def __str__(self):
        return self.smarts

    def __repr__(self):
        return self.__str__()

    def find_wildcard_positions(self, smarts_mol):
        # 任意の原子 ([!#1]) に該当する原子のインデックスを格納するリスト
        wildcard_indices = []

        # 各原子を調べる
        for atom in smarts_mol.GetAtoms():
            # SMARTSパターンにおいて任意の原子 ([!#1]) に該当する場合
            # if atom.HasProp("_QueryAtomFeatures") and '[!#1]' in atom.GetSmarts():
            if '[!#1]' in atom.GetSmarts():
                wildcard_indices.append(atom.GetIdx())

        return wildcard_indices
    
    def match(self, mol):
        return [
            tuple(-1 if i in self.wildcard_indices else atom for i, atom in enumerate(_match))
            for _match in mol.GetSubstructMatches(self.mol)
        ]
            
        
        
class Fragmentizer:
    func_group = {
        '[!#1]-[OH]':{},
        '[!#1]=O':{
            '[!#1]-N=O':{},
        },
        '[!#1]-[CX3H1](=O)':{},
        '[!#1]-O-[!#1]' : {},
        '[!#1]-C(=O)-O-[!#1]' : {},

        '[!#1]-[NH2]':{},
        '[!#1]=[NH]':{},
        '[!#1]-[N+](=O)[O-]':{},
        '[!#1]-C#N':{},
        '[!#1]-N=N-[!#1]':{},
        '[!#1]-[NH]-[NH]-[!#1]':{},
        '[!#1]-[C;X3](=O)(-[N;X2,X3H1,X3H0]-,=[!#1])':{
            '[!#1]-[C;X3](=O)(-[NH1]-[!#1])':{},
            '[!#1]-[C;X3](=O)(-[N;X2]=[!#1])':{},
            '[!#1]-[C;X3](=O)(-[N;X3](-[!#1])-[!#1])':{},
        },

        
        '[!#1]-[SH1]' : {},
        '[!#1]=S' : {},
        '[!#1]-[S](=O)(=O)':{},
        '[!#1]-[S](=O)(=O)(-[OH])':{},
        '[!#1]-[S](=O)(=O)-[!#1]' : {},
        '[!#1]-S-[!#1]' : {},


        '[!#1]-[P](=O)(-O)(-O)':{
            '[!#1]-O-[P](=O)(-O)(-O)':{},
        },

        '[!#1]-[F,Cl,Br,I]' : {
            '[!#1]-F' : {},
            '[!#1]-Cl' : {},
            '[!#1]-Br' : {},
            '[!#1]-I' : {},
        },

    }

    def __init__(self, max_attach_atom_cnt):
        self.max_attach_atom_cnt = max_attach_atom_cnt
        all_query = {}
        root_query_idx = []
        query_parent_idx = {}
        query_child_idx = {}
        idx = -1
        
        def traverse_and_build(parent_idx, children):
            nonlocal idx
            brother_indices = []
            for smarts, grandchildren in children.items():
                fg = FunctionalGroup(smarts)
                idx += 1
                current_idx = idx
                all_query[current_idx] = fg
                query_parent_idx[current_idx] = parent_idx
                brother_indices.append(current_idx)
                query_child_idx[current_idx] = traverse_and_build(current_idx, grandchildren)
            return brother_indices

        for smarts, children in self.func_group.items():
            idx += 1
            current_idx = idx
            fg = FunctionalGroup(smarts)
            all_query[current_idx] = fg
            query_parent_idx[current_idx] = -1
            root_query_idx.append(current_idx)
            query_child_idx[current_idx] = traverse_and_build(current_idx, children)
        
        self.all_query:dict[int,FunctionalGroup] = all_query
        self.root_query_idx = root_query_idx
        self.query_parent_idx = query_parent_idx
        self.query_child_idx = query_child_idx


    def split_to_motif(self, mol) -> Tuple[List[Motif], FragmentGroup]:
        fragment = Fragment(mol)
        max_attach_atom_cnt = self.max_attach_atom_cnt

        fg_list = self.assign_functional_group(fragment) # [(match, fg_query_idx), ...]
        all_fg_indices = set([i for match, _ in fg_list for i in match])

        fragment_group, ring_indices = self.sep_ring(fragment)

        motif_atom_map_group = []
        attachment_atom_map_dict = {}
        chain_atom_map = []
        seps = []
        for ring_i in ring_indices:
            motif_atom_map_group.append(fragment_group[ring_i].atom_map)
            attachment_atom_map_dict[len(motif_atom_map_group)-1] = {}
            neighbor_infoes = fragment_group.get_neighbors(ring_i)
            for s_bond_pos, (e_frag_idx, e_bond_pos) in neighbor_infoes.items():
                next_infoes = [(ring_i, s_bond_pos, e_frag_idx, e_bond_pos)]
                visited_atom_map = set()
                is_connection = False
                while len(next_infoes) > 0:
                    pre_s_frag_idx, pre_s_bond_pos, pre_e_frag_idx, pre_e_bond_pos = next_infoes.pop(0)
                    frag = fragment_group[pre_e_frag_idx]
                    if frag.atom_map[frag.bond_list[pre_e_bond_pos].atom_idx] in visited_atom_map:
                        continue
                    if frag.id in ring_indices:
                        if len(visited_atom_map) > 0:
                            is_connection = True
                        else:
                            if len(visited_atom_map) == 0:
                                atom_map1 = fragment_group[ring_i].atom_map[fragment_group[ring_i].bond_list[s_bond_pos].atom_idx]
                                atom_idx1 = fragment.atom_map.index(atom_map1)
                                atom_map2 = fragment_group[e_frag_idx].atom_map[fragment_group[e_frag_idx].bond_list[e_bond_pos].atom_idx]
                                atom_idx2 = fragment.atom_map.index(atom_map2)
                                if atom_idx1 < atom_idx2:
                                    seps.append((atom_idx1, atom_idx2))
                                else:
                                    seps.append((atom_idx2, atom_idx1))
                        continue
                    visited_atom_map.update(frag.atom_map)

                    for s_bond_pos2, (e_frag_idx2, e_bond_pos2) in fragment_group.get_neighbors(pre_e_frag_idx).items():
                        neighbor_frag = fragment_group[e_frag_idx2]
                        if s_bond_pos2 == pre_e_bond_pos:
                            continue

                        next_infoes.append((pre_e_frag_idx, s_bond_pos2, e_frag_idx2, e_bond_pos2))
                
                if is_connection or len(visited_atom_map) > max_attach_atom_cnt:
                    chain_atom_map.append(tuple(visited_atom_map))
                    atom_idx = fragment_group[ring_i].bond_list[s_bond_pos].atom_idx
                    a_map = fragment_group[ring_i].atom_map[atom_idx]
                    if a_map not in attachment_atom_map_dict[len(motif_atom_map_group)-1]:
                        attachment_atom_map_dict[len(motif_atom_map_group)-1][a_map] = []
                    attachment_atom_map_dict[len(motif_atom_map_group)-1][a_map].append(())

                else:
                    atom_idx = fragment_group[ring_i].bond_list[s_bond_pos].atom_idx
                    a_map = fragment_group[ring_i].atom_map[atom_idx]
                    if a_map not in attachment_atom_map_dict[len(motif_atom_map_group)-1]:
                        attachment_atom_map_dict[len(motif_atom_map_group)-1][a_map] = []
                    attachment_atom_map_dict[len(motif_atom_map_group)-1][a_map].append(tuple(visited_atom_map))
        if len(ring_indices) == 0:
            chain_atom_map = [fragment.atom_map]
        else:
            chain_atom_map = list(set(chain_atom_map))


        chain_atom_idx = [[fragment.atom_map.index(i) for i in chain] for chain in chain_atom_map]
        for atom_indices in chain_atom_idx:
            next_idx = [atom_indices[0]]
            visited_indices = set([atom_indices[0]])
            while len(next_idx) > 0:
                current_atom = fragment.mol.GetAtomWithIdx(next_idx.pop(0))
                atom_idx = current_atom.GetIdx()
                for neighbor in current_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in visited_indices:
                        continue
                    visited_indices.add(neighbor_idx)
                    
                    if neighbor_idx not in atom_indices:
                        if atom_idx < neighbor_idx:
                            seps.append((atom_idx, neighbor_idx))
                        else:
                            seps.append((neighbor_idx, atom_idx))
                        continue
                    
                    next_idx.append(neighbor_idx)

                    if atom_idx in all_fg_indices and neighbor_idx in all_fg_indices:
                        continue

                    bond = fragment.mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                        continue

                    if atom_idx < neighbor_idx:
                        seps.append((atom_idx, neighbor_idx))
                    else:
                        seps.append((neighbor_idx, atom_idx))
                    
        fragment_group2 = split_fragment(fragment, list(set(seps)))

        motifs = []
        for frag in fragment_group2:
            if frag.mol.GetRingInfo().NumRings() == 0:
                motif = Motif.from_fragment(frag, seps=[], motif_atom_map=frag.atom_map)
                motifs.append(motif)
            else:
                for i, motif_atom_map in enumerate(motif_atom_map_group):
                    if not set(motif_atom_map).issubset(set(frag.atom_map)):
                        continue
                    seps = []
                    attachment_atom_map = attachment_atom_map_dict[i]
                    for s_a_map, attach_maps in attachment_atom_map.items():
                        for attach_map in attach_maps:
                            for e_a_map in attach_map:
                                atom_idx1 = frag.atom_map.index(s_a_map)
                                atom_idx2 = frag.atom_map.index(e_a_map)
                                bond = frag.mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
                                if bond is not None:
                                    if atom_idx1 < atom_idx2:
                                        seps.append((atom_idx1, atom_idx2))
                                    else:
                                        seps.append((atom_idx2, atom_idx1))
                                    break
                    motif = Motif.from_fragment(frag, seps=seps, motif_atom_map=motif_atom_map)
                    motifs.append(motif)
                    
                    break
                else:
                    raise ValueError(f"Ring fragment does not contain any motif.\t{frag}")

        return motifs, fragment_group2
                


    def assign_functional_group(self, fragment):
        """
        Assigns functional groups to the fragment.

        Args:
            fragment (Fragment): Fragment object containing the molecule.

        Returns:
            FragmentGroup: A group of fragments split by the functional groups.
        """
        fg_candidates = []
        for query_idx, query in self.all_query.items():
            matches = query.match(fragment.mol)
            
            for match in matches:
                fg_candidates.append((tuple(i for i in match if i != -1), query_idx, query.num_atoms))

        fg_candidates.sort(key=lambda x: x[2], reverse=True)
        fg_confirmed = []
        
        while len(fg_candidates) > 0:
            match, query_idx, num_atoms = fg_candidates.pop(0)
            fg_confirmed.append((match, query_idx))
            for i in range(len(fg_candidates)-1, -1, -1):
                if set(fg_candidates[i][0]) & set(match):
                    del fg_candidates[i]

        return fg_confirmed

        
    def sep_ring(self, fragment, allowed_bond_types=None):
        """
        Separates a fragment into smaller fragments based on non-ring bonds and identifies fragments containing rings.

        This function identifies bonds in a fragment that are not part of any ring structure and filters the bonds 
        based on their bond type (allowed_bond_types). It then splits the fragment into smaller fragments at those bonds. 
        It also determines which of the resulting fragments contain ring structures.

        Args:
            fragment (Fragment): The input fragment to be processed. It should have a `mol` attribute that is an RDKit molecule object.
            allowed_bond_types (set[Chem.BondType], optional): A set of RDKit bond types that are allowed to be cut. 
                Defaults to {Chem.BondType.SINGLE}, meaning only single bonds are cut.

        Returns:
            tuple:
                - fragment_group (FragmentGroup): A group of fragments obtained after splitting the input fragment.
                - ring_indices (list[int]): A list of fragment IDs corresponding to fragments that contain ring structures.
        """
        # By default, only single bonds are allowed to be cut
        if allowed_bond_types is None:
            allowed_bond_types = {Chem.rdchem.BondType.SINGLE}
        
        # Retrieve groups of atoms that form ring structures
        ring_groups = get_ring_groups(fragment.mol)

        # Set to store bonds that are not part of the ring
        seps = set()

        # Iterate through the atoms in the ring structures
        for ring_group in ring_groups:
            for atom_idx in ring_group:
                atom = fragment.mol.GetAtomWithIdx(atom_idx)  # Retrieve the atom

                # Check all bonds connected to the atom
                for bond in atom.GetBonds():
                    start_idx = bond.GetBeginAtomIdx()  # Start atom index of the bond
                    end_idx = bond.GetEndAtomIdx()     # End atom index of the bond

                    # Identify bonds that are not part of the ring
                    if not bond.IsInRing():
                        if bond.GetBondType() in allowed_bond_types:
                            if start_idx < end_idx:
                                seps.add((start_idx, end_idx))
                            else:
                                seps.add((end_idx, start_idx))
                        else:
                            visited_atom_indices = set([start_idx, end_idx])
                            visited_atom_indices.update(ring_group)
                            next_indices = [start_idx, end_idx]
                            joint_cnt = 0
                            while len(next_indices) > 0:
                                current_idx = next_indices.pop(0)
                                atom = fragment.mol.GetAtomWithIdx(current_idx)
                                for neighbor in atom.GetNeighbors():
                                    neighbor_idx = neighbor.GetIdx()
                                    if neighbor_idx in visited_atom_indices:
                                        continue
                                    visited_atom_indices.add(neighbor_idx)
                                    
                                    bond = fragment.mol.GetBondBetweenAtoms(current_idx, neighbor_idx)
                                    if bond.IsInRing():
                                        joint_cnt += 1
                                        continue
                                    if bond.GetBondType() not in allowed_bond_types:
                                        next_indices.append(neighbor_idx)
                                        continue

                                    if current_idx < neighbor_idx:
                                        seps.add((current_idx, neighbor_idx))
                                    else:
                                        seps.add((neighbor_idx, current_idx))
                                    joint_cnt += 1

                            if joint_cnt == 0:
                                if start_idx < end_idx:
                                    seps.add((start_idx, end_idx))
                                else:
                                    seps.add((end_idx, start_idx))
        
        # Split the fragment using the identified non-ring bonds
        fragment_group = split_fragment(fragment, list(seps))
        
        ring_indices = []
        for frag in fragment_group:
            if frag.mol.GetRingInfo().NumRings() > 0:
                ring_indices.append(frag.id)
        
        return fragment_group, ring_indices
                

    def _find_group(self, groups, idx):
        for group_index, group in enumerate(groups):
            if idx in group:
                return group_index
        return -1 
    

            

    