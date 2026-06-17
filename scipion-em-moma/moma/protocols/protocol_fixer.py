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

"""
Reconstructs backbone atoms (N, C, O) from a Cα-only ensemble using ideal backbone geometry. This is needed to use Cα-only ensembles-.
"""
import os
from pyworkflow.constants import BETA
import numpy as np
from pwem.protocols import EMProtocol
from pwem.objects import AtomStruct, SetOfAtomStructs
from pyworkflow.protocol import params
from pyworkflow.utils import Message

class ProtocolFixer(EMProtocol):
    """
    This protocol will reconstruct a protein file containing only Ca atoms from a given pdb file. Will add CB and N atoms. 
    
    """
    _label = 'Fixer'
    _devStatus = BETA
    
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """

        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputEnsemble', params.PointerParam,
                      pointerClass='SetOfAtomStructs',
                      label='Input ensemble (Cα-only)',
                      important=True,
                      help='SetOfAtomStructs with one Cα per residue, '
                           'e.g. output from ANM MC walks.')

        form.addParam('useReference', params.BooleanParam,
                      default=False,
                      label='Validate with reference structure?',
                      help='If Yes, provide an all-atom PDB to validate '
                           'the reconstruction quality.')

        form.addParam('referenceStructure', params.PointerParam,
                      pointerClass='AtomStruct',
                      label='Reference structure (all-atom)',
                      condition='useReference',
                      help='Original all-atom structure (.pdb or .cif). Used to compare '
                        'reconstructed N and C positions against real coordinates '
                        'to assess the quality of the backbone reconstruction.')

        
    def _insertAllSteps(self):
        self._insertFunctionStep(self.reconstructStep, needsGPU=False)   
        self._insertFunctionStep(self.validateStep, needsGPU=False)
        self._insertFunctionStep(self.createOutputStep, needsGPU=False)



    def reconstructStep(self):
        ensemble = self.inputEnsemble.get()
        out_dir = self._getExtraPath('reconstructed')
        os.makedirs(out_dir, exist_ok=True)

        for i, atom_struct in enumerate(ensemble):
            src = atom_struct.getFileName()
            dst = os.path.join(out_dir, f'frame_{i:04d}.pdb')
            self._reconstructBackbone(src, dst)
            self._log.info(f'Reconstructed frame {i:04d}')


    def validateStep(self):
        """Validate reconstruction quality using the reference structure."""

        if not self.useReference.get():
            print(">>> Validation skipped: no reference structure provided.")
            return

        out_dir = self._getExtraPath('reconstructed')
        pdb_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.pdb')])

        if not pdb_files:
            print(">>> No reconstructed frames found for validation.")
            return

        # Convertir referencia si es mmCIF
        reference_raw = self.referenceStructure.get().getFileName()
        if reference_raw.endswith('.cif'):
            reference_all_atom = self._getExtraPath('reference.pdb')
            self._convertCifToPdb(reference_raw, reference_all_atom)
        else:
            reference_all_atom = reference_raw

        # Extract Ca
        reference_ca = self._getExtraPath('reference_ca.pdb')
        self._extractCaFromPdb(reference_all_atom, reference_ca)

        # Rebould backbone from Ca
        reference_rebuilt = self._getExtraPath('reference_rebuilt.pdb')
        self._reconstructBackbone(reference_ca, reference_rebuilt)

        # Compare reconstructed backbone to original all-atom reference
        recon_atoms = self._parseBackbone(reference_rebuilt)
        ref_atoms   = self._parseBackbone(reference_all_atom)

        common = set(recon_atoms) & set(ref_atoms)
        N_errors, C_errors = [], []

        for key in common:
            _, _, atom = key
            err = np.linalg.norm(
                np.array(recon_atoms[key]) - np.array(ref_atoms[key])
            )
            if atom == 'N':
                N_errors.append(err)
            elif atom == 'C':
                C_errors.append(err)

        print('=== Backbone reconstruction validation ===')
        print(f'Reference: {reference_all_atom}')
        if N_errors:
            print(f'N error: {np.mean(N_errors):.2f} +/- '
                f'{np.std(N_errors):.2f} A  '
                f'(max: {np.max(N_errors):.2f} A)')
        if C_errors:
            print(f'C error: {np.mean(C_errors):.2f} +/- '
                f'{np.std(C_errors):.2f} A  '
                f'(max: {np.max(C_errors):.2f} A)')

        mean_N = np.mean(N_errors) if N_errors else 0
        mean_C = np.mean(C_errors) if C_errors else 0
        if mean_N > 3.0 or mean_C > 3.0:
            print('WARNING: error > 3 A. Ideal geometry may not be accurate enough.')
        else:
            print('OK: reconstruction quality within acceptable range.')

    def _extractCaFromPdb(self, input_pdb, output_ca_pdb):
        """Extract only CA atoms from an all-atom PDB."""
        with open(input_pdb, 'r') as f_in, open(output_ca_pdb, 'w') as f_out:
            for line in f_in:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    f_out.write(line)
            f_out.write('END\n')

    def createOutputStep(self):
        out_dir = self._getExtraPath('reconstructed')
        pdb_files = sorted([
            f for f in os.listdir(out_dir) if f.endswith('.pdb')
        ])

        output_set = SetOfAtomStructs.create(self._getPath())

        for pdb_file in pdb_files:
            pdb_path = os.path.join(out_dir, pdb_file)
            atom_struct = AtomStruct()
            atom_struct.setFileName(pdb_path)
            output_set.append(atom_struct)

        self._defineOutputs(outputStructures=output_set)
        self._defineSourceRelation(self.inputEnsemble, output_set)

    def _reconstructBackbone(self, ca_pdb, output_pdb):
        ca_coords, residues, chains, resnames = self._parseCaOnly(ca_pdb)
        records = []
        atom_serial = 1

        for chain_id in np.unique(chains):
            mask = chains == chain_id
            ca = ca_coords[mask]
            res = residues[mask]
            rnames = resnames[mask]
            n_res = len(ca)

            N_pos = np.zeros_like(ca)
            C_pos = np.zeros_like(ca)
            CB_pos = np.zeros_like(ca)

            for i in range(n_res):
                if i == 0:
                    forward = ca[1] - ca[0]
                elif i == n_res - 1:
                    forward = ca[-1] - ca[-2]
                else:
                    forward = ca[i + 1] - ca[i - 1]

                forward = forward / np.linalg.norm(forward)

                perp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(forward, perp)) > 0.9:
                    perp = np.array([0.0, 1.0, 0.0])
                perp = perp - np.dot(perp, forward) * forward
                perp = perp / np.linalg.norm(perp)

                angle_rad = np.radians(54.6)
                N_dir = (-np.cos(angle_rad) * forward +
                        np.sin(angle_rad) * perp)
                N_pos[i] = ca[i] + 1.458 * N_dir

                C_dir = (np.cos(angle_rad) * forward +
                        np.sin(angle_rad) * perp)
                C_pos[i] = ca[i] + 1.525 * C_dir

                n_dir = N_pos[i] - ca[i]
                n_dir = n_dir / np.linalg.norm(n_dir)
                c_dir = C_pos[i] - ca[i]
                c_dir = c_dir / np.linalg.norm(c_dir)

                bisect = n_dir + c_dir
                if np.linalg.norm(bisect) > 1e-6:
                    bisect = bisect / np.linalg.norm(bisect)
             
                normal = np.cross(n_dir, c_dir)
                if np.linalg.norm(normal) > 1e-6:
                    normal = normal / np.linalg.norm(normal)

                cb_angle = np.radians(54.7)
                CB_dir = (-np.cos(cb_angle) * bisect +
                        np.sin(cb_angle) * normal)
                CB_dir = CB_dir / np.linalg.norm(CB_dir)
                CB_pos[i] = ca[i] + 1.521 * CB_dir

            for i in range(n_res):
                res_id = res[i]
                rname = rnames[i]

                records.append(self._pdb_line(
                    atom_serial, 'N', rname, chain_id, res_id, N_pos[i]))
                atom_serial += 1

                records.append(self._pdb_line(
                    atom_serial, 'CA', rname, chain_id, res_id, ca[i]))
                atom_serial += 1

                records.append(self._pdb_line(
                    atom_serial, 'C', rname, chain_id, res_id, C_pos[i]))
                atom_serial += 1

                #CB only for non-glycine residues
                if rname != 'GLY':
                    records.append(self._pdb_line(
                        atom_serial, 'CB', rname, chain_id, res_id, CB_pos[i]))
                    atom_serial += 1

            records.append(f'TER   {atom_serial:5d}      '
                        f'{rnames[-1]:3s} {chain_id}{res[-1]:4d}\n')
            atom_serial += 1

        records.append('END\n')

        with open(output_pdb, 'w') as f:
            f.writelines(records)

    def _parseCaOnly(self, pdb_path):
        """Parse a Cα-only PDB and return arrays."""
        coords, residues, chains, resnames = [], [], [], []

        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                resname = line[17:20].strip()
                chain = line[21].strip()
                resid = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                coords.append([x, y, z])
                residues.append(resid)
                chains.append(chain)
                resnames.append(resname)

        return (np.array(coords),
                np.array(residues),
                np.array(chains),
                np.array(resnames))
    
    def _parseBackbone(self, pdb_path):
        """Parse N, CA, C coords from a PDB. Returns dict {(chain,resid,atom): coord}."""
        atoms = {}
        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                atom = line[12:16].strip()
                if atom not in ('N', 'CA', 'C'):
                    continue
                chain = line[21]
                resid = int(line[22:26])
                coord = (float(line[30:38]),
                         float(line[38:46]),
                         float(line[46:54]))
                atoms[(chain, resid, atom)] = coord
        return atoms

    def _pdb_line(self,serial, atom_name, resname, chain, resid, coord):
        """Format a PDB ATOM record line."""
        name_field = f' {atom_name:<3s}' if len(atom_name) < 4 else atom_name
        return (f'ATOM  {serial:5d} {name_field} {resname:3s} '
                f'{chain}{resid:4d}    '
                f'{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}'
                f'  1.00  0.00           {atom_name[0]:>2s}\n')
    
    def _convertCifToPdb(self, cif_path, pdb_path):
        from Bio.PDB import MMCIFParser, PDBIO
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('ref', cif_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)
        return pdb_path
