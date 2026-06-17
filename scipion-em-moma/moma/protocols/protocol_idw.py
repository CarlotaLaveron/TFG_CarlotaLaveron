# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Carlota Laverón (carlota.laveronvilas@usp.ceu.es)
# *
# * CEU San Pablo University
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'carlota.laveronvilas@usp.ceu.es'
# *
# **************************************************************************

import os
import copy

import numpy as np
from scipy.spatial import cKDTree

from pyworkflow.constants import BETA
from pyworkflow.protocol import params
from pwem.protocols import EMProtocol
from pwem.objects import SetOfAtomStructs, AtomStruct

from ..objects import InvDistTree3D


DEFAULT_SEARCH_RADIUS = 15.0
DEFAULT_IDW_POWER = 2.0
OUTPUT_STRUCTURES_NAME = 'reconstructedStructures'


class ProtocolInverseDistanceWeighting(EMProtocol):
    """
    Reconstruct full-atom protein structures from a Ca-only ensemble using
    compact-support Inverse Distance Weighting (IDW) + KD-Tree.

    Input:  AtomStruct (full-atom reference) + SetOfAtomStructs (Ca-only ensemble).
    Output: SetOfAtomStructs with reconstructed full-atom structures.
    """

    _label = 'IDW Backbone Reconstruction'
    _devStatus = BETA

    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam(
            'inputReference',
            params.PointerParam,
            pointerClass='AtomStruct',
            label='Reference full-atom structure',
            important=True,
            help='PDB/CIF file containing ALL atoms. Its C_alfa positions '
                 'serve as source control points for the IDW interpolation.',
        )

        form.addParam(
            'inputEnsemble',
            params.PointerParam,
            pointerClass='SetOfAtomStructs',
            label='Target C_alfa ensemble',
            important=True,
            help='Set of C_alfa-only models (one per conformer), e.g. from '
                 'prody2 - ANM MC walks. Each model provides destination '
                 'C_alfa positions.',
        )

        form.addSection(label='IDW parameters')

        form.addParam(
            'searchRadius',
            params.FloatParam,
            default=DEFAULT_SEARCH_RADIUS,
            label='Search radius R (A)',
            help='Maximum distance in Angstroms to search for C_alfa neighbours. '
                 'Ca beyond this radius contribute exactly 0. '
                 'Typical values: 10-20 A.\n\n'
                 'Formula: w_k = ((R - d) / (R * d)) ^ p',
        )

    def _insertAllSteps(self):
        self._insertFunctionStep('reconstructStep')
        self._insertFunctionStep('createOutputStep')

    def reconstructStep(self):
        from Bio.PDB import PDBIO
        R = self.searchRadius.get()

        ref_path = self.inputReference.get().getFileName()
        ref_struct = self._parse_structure(ref_path, 'reference')


        src_ca_coords, src_ca_keys = self.extract_ca_coords(ref_struct)
        all_atom_coords, all_atoms = self.extract_all_atom_coords(ref_struct)

        if src_ca_coords.shape[0] == 0:
            raise RuntimeError(
                'No C_alfa atoms found in reference structure: %s' % ref_path
            )

        # Automatic k selection based on mean number of C_alfa within R
        tree_ref = cKDTree(src_ca_coords)
        counts = [len(tree_ref.query_ball_point(ca, r=R)) for ca in src_ca_coords]
        k = max(int(np.mean(counts) * 2), 1)

        self.info('Reference: %d C_alfa, %d total atoms, R=%.1f A, k=%d (auto), p=%.2f'
                  % (len(src_ca_keys), len(all_atoms), R, k, DEFAULT_IDW_POWER))
        self.info('Auto k = %d (mean C_alfa within R=%.1f A: %.1f)'
                  % (k, R, np.mean(counts)))

        out_dir = self._getExtraPath('reconstructed')
        os.makedirs(out_dir, exist_ok=True)

        ensemble = self.inputEnsemble.get()
        n_models = len(ensemble)
        self.info('Ensemble: %d model(s).' % n_models)

        io = PDBIO()

        idw = InvDistTree3D(src_ca_coords, leafsize=10)

        for i, target_as in enumerate(ensemble):
            target_path = target_as.getFileName()
            target_struct = self._parse_structure(target_path, 'target_%d' % i)
            dst_ca_coords, dst_ca_keys = self.extract_ca_coords(target_struct)

            src_aligned, dst_aligned = self._align_ca_by_key(
                src_ca_coords, src_ca_keys,
                dst_ca_coords, dst_ca_keys,
            )

            if src_aligned.shape[0] == 0:
                self.warning('No matching C_alfa for model %d - skipping.' % i)
                continue

            self.info('[%d/%d] %d matched C_alfa pairs'
                      % (i + 1, n_models, src_aligned.shape[0]))

            new_coords = self.reconstruct_atoms(
                src_ca = src_aligned,
                dst_ca = dst_aligned,
                all_atoms = all_atom_coords,
                idw = idw,
                R = R,
                k = k,
            )


            out_struct = copy.deepcopy(ref_struct)
            _, out_atoms = self.extract_all_atom_coords(out_struct)
            self.apply_coords_to_structure(out_atoms, new_coords)

            out_path = os.path.join(out_dir, 'reconstructed_%04d.pdb' % i)
            io.set_structure(out_struct)
            io.save(out_path)
            self._fix_pdb_format(out_path)

        self.info('Reconstruction complete.')


    def createOutputStep(self):
        out_dir   = self._getExtraPath('reconstructed')
        pdb_files = sorted(f for f in os.listdir(out_dir) if f.endswith('.pdb'))

        if not pdb_files:
            raise RuntimeError('No reconstructed PDB files in %s' % out_dir)

        output_set = SetOfAtomStructs.create(self._getPath())

        for pdb_file in pdb_files:
            as_obj = AtomStruct()
            as_obj.setFileName(os.path.join(out_dir, pdb_file))
            output_set.append(as_obj)

        self._defineOutputs(**{OUTPUT_STRUCTURES_NAME: output_set})
        self._defineSourceRelation(self.inputReference, output_set)
        self._defineSourceRelation(self.inputEnsemble,  output_set)
        self.info('Output: %d structure(s).' % len(output_set))

    def _parse_structure(self, path, struct_id):
        from Bio.PDB import PDBParser, MMCIFParser
        ext = os.path.splitext(path)[1].lower()
        parser = MMCIFParser(QUIET=True) if ext in ('.cif', '.mmcif') \
                 else PDBParser(QUIET=True)
        return parser.get_structure(struct_id, path)

    @staticmethod
    def _align_ca_by_key(src_coords, src_keys, dst_coords, dst_keys):
        dst_map = dict(zip(dst_keys, dst_coords))
        src_aligned, dst_aligned = [], []
        for key, coord in zip(src_keys, src_coords):
            if key in dst_map:
                src_aligned.append(coord)
                dst_aligned.append(dst_map[key])

        missing = [k for k in src_keys if k not in dst_map]
        if missing:
            print('WARNING: %d Ca in reference have no match in target: %s'
                % (len(missing), missing[:5]))

        if not src_aligned:
            return np.empty((0, 3)), np.empty((0, 3))
        return np.array(src_aligned), np.array(dst_aligned)
    
    @staticmethod
    def _count_no_neighbour(all_atoms, src_ca, R):
        tree = cKDTree(src_ca)
        hits = tree.query_ball_point(all_atoms, r=R, workers=-1)
        return sum(1 for h in hits if len(h) == 0)

    def reconstruct_atoms(self, src_ca, dst_ca, all_atoms, idw, 
                          R=15.0, k=8, power=2.0, leafsize=10):
        src_ca    = np.asarray(src_ca,    dtype=np.float64)
        dst_ca    = np.asarray(dst_ca,    dtype=np.float64)
        all_atoms = np.asarray(all_atoms, dtype=np.float64)

        if src_ca.shape != dst_ca.shape:
            raise ValueError(
                "src_ca and dst_ca must have the same shape, "
                "got %s vs %s" % (src_ca.shape, dst_ca.shape)
            )

        displacements = np.asarray(dst_ca, dtype=np.float64) - np.asarray(src_ca, dtype=np.float64)
        interpolated  = idw(all_atoms, displacements, R=R, k=k)
        return all_atoms + interpolated

    def extract_ca_coords(self, structure):
        coords, keys = [], []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    if 'CA' not in residue:
                        continue
                    coords.append(residue['CA'].get_vector().get_array())
                    keys.append((chain.id, residue.id[1], residue.id[2].strip()))
            break
        return np.array(coords, dtype=np.float64), keys

    def extract_all_atom_coords(self, structure):
        coords, atoms = [], []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    for atom in residue:
                        coords.append(atom.get_vector().get_array())
                        atoms.append(atom)
            break
        return np.array(coords, dtype=np.float64), atoms

    def apply_coords_to_structure(self, atoms, new_coords):
        for atom, coord in zip(atoms, new_coords):
            atom.set_coord(coord)

    def _summary(self):
        out = getattr(self, OUTPUT_STRUCTURES_NAME, None)
        if self.isFinished() and out is not None:
            return [
                'Reconstructed %d full-atom structure(s).' % len(out),
                'R=%.1f A, k=auto (mean Ca within R * 2), p=2.0 (fixed).' % self.searchRadius.get(),
                'Check the log for the exact k value used.',
            ]
        return ['Protocol not finished yet.']

    def _methods(self):
        return [
            'Full-atom structures were reconstructed from a Ca-only ensemble '
            'using compact-support IDW with KD-Tree neighbour search '
            '(R=%.1f A, p=%.2f). '
            'Formula: w_k = ((R - d(x,x_k)) / (R * d(x,x_k)))^p.' % (
                self.searchRadius.get(), DEFAULT_IDW_POWER)
        ]

    def _validate(self):
        errors = []
        if self.inputReference.get() is None:
            errors.append('A reference full-atom structure must be provided.')
        ens = self.inputEnsemble.get()
        if ens is None:
            errors.append('A target Ca ensemble must be provided.')
        elif len(ens) == 0:
            errors.append('The target ensemble is empty.')
        if self.searchRadius.get() <= 0:
            errors.append('Search radius R must be positive.')
        return errors

    def _createSetOfAtomStructs(self):
        return SetOfAtomStructs.create(self._getPath())

    @staticmethod
    def _fix_pdb_format(pdb_path):
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        fixed      = []
        atom_count = 0
        last_atom  = None
        in_hetatm  = False

        for line in lines:
            record = line[:6].strip()

            if record == 'ATOM':
                in_hetatm  = False
                atom_count += 1
                last_atom  = line
                fixed.append(line)

            elif record == 'HETATM':
                if not in_hetatm and last_atom is not None:
                    atom_count += 1
                    res_name = last_atom[17:20]
                    chain    = last_atom[21]
                    resseq   = last_atom[22:26]
                    fixed.append('TER   %5d      %3s %s%4s\n' % (
                        atom_count, res_name, chain, resseq
                    ))
                    in_hetatm = True
                atom_count += 1
                fixed.append(line)

            elif record in ('TER', 'END'):
                continue

            else:
                fixed.append(line)

        if not in_hetatm and last_atom is not None:
            atom_count += 1
            res_name = last_atom[17:20]
            chain    = last_atom[21]
            resseq   = last_atom[22:26]
            fixed.append('TER   %5d      %3s %s%4s\n' % (
                atom_count, res_name, chain, resseq
            ))

        fixed.append('END\n')

        with open(pdb_path, 'w') as f:
            f.writelines(fixed)