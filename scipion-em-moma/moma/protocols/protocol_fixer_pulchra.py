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

##NOT PART OF THE TFG

"""
Reconstructs backbone atoms (N, C, O) from a Cα-only ensemble using ideal backbone geometry. This is needed to use Cα-only ensembles-.
"""
import os
import shutil
import subprocess
import numpy as np
import tempfile
import moma


from moma.constants import MOMA_DIC
from pwem.protocols import EMProtocol
from pwem.objects import AtomStruct, SetOfAtomStructs
from pyworkflow import BETA
from moma.pulchra_installer import PulchraInstaller
from pwchem import Plugin as pwchemPlugin
from pyworkflow.protocol import params
from pyworkflow.utils import Message


class ProtocolFixerPulchra(EMProtocol):
    """
    Reconstructs full backbone (N, CA, C, O, CB) from Cα-only PDB files
    using PULCHRA.
    """
    _label = 'Fixer Pulchra'
    _devStatus = BETA


    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputEnsemble', params.PointerParam,
                      pointerClass='SetOfAtomStructs',
                      label='Input ensemble (Cα-only)',
                      important=True,
                      help='SetOfAtomStructs with one Cα per residue, '
                           'e.g. output from ANM MC walks.')

    def _insertAllSteps(self):
        self._insertFunctionStep(self.reconstructStep, needsGPU=False)
        self._insertFunctionStep(self.refineStep, needsGPU=False)
        self._insertFunctionStep(self.createOutputStep, needsGPU=False)

    def reconstructStep(self):
        ensemble = self.inputEnsemble.get()
        out_dir = self._getExtraPath('reconstructed')
        os.makedirs(out_dir, exist_ok=True)

        structs = list(ensemble)
        total = len(structs)

        pulchra_bin = PulchraInstaller.getPulchraBin()
        if not PulchraInstaller.pulchraExists():
            raise FileNotFoundError(
                'PULCHRA binary not found. '
                'Run:  scipion3 installb pulchra  '
                'or check PULCHRA_HOME in your Scipion config.'
            )

        for i, atom_struct in enumerate(ensemble):
            src = atom_struct.getFileName()
            dst = os.path.join(out_dir, f'frame_{i:04d}.pdb')
            pct = (i + 1) / total * 100
            self.info(f'[Reconstruct] Frame {i+1}/{total} ({pct:.1f}%) - {os.path.basename(src)}')

            self.runPulchra_allchains(pulchra_bin, src, dst)


    def refineStep(self):
        out_dir     = self._getExtraPath('reconstructed')
        refined_dir = self._getExtraPath('refined')
        os.makedirs(refined_dir, exist_ok=True)

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        args = f'--input "{os.path.abspath(out_dir)}" --output "{os.path.abspath(refined_dir)}"'

        pwchemPlugin.runScript(
            self,
            'refine_runner.py',
            args,
            env=MOMA_DIC,
            cwd=refined_dir,
            scriptDir=runner_dir,
        )


    def _prepare_chain_for_pulchra(self, lines):
        STANDARD_AA = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL',
            # modificados que PULCHRA reconoce explícitamente
            'HID', 'ASX', 'GLX', 'TPO', 'MSE'
            }
        """
        Cleans a list of PDB lines for Pulchra processig:
        1. Deletes no-A altlocs
        2. Normaliza altloc A -> espacio
        3. Filtra solo ATOM de proteína estándar
        4. Filtra solo CA (si el input es CA-only, no hace nada malo)
        """
        cleaned = []
        for line in lines:
            if not line.startswith("ATOM"):
                continue

            # Filtrar altlocs: saltar B, C, D...
            if len(line) > 16 and line[16] not in (' ', 'A'):
                continue

            # Normalizar altloc A -> espacio
            if len(line) > 16 and line[16] == 'A':
                line = line[:16] + ' ' + line[17:]

            # Filtrar solo aminoácidos que PULCHRA entiende
            resname = line[17:20].strip()
            if resname not in STANDARD_AA:
                continue

            cleaned.append(line)
        return cleaned

    def _split_by_chain(self, pdb_path):
        chains = {}
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and len(line) > 21:
                    chain_id = line[21]
                    chains.setdefault(chain_id, []).append(line)
        return chains

    def runPulchra_allchains(self, pulchra_bin, ca_pdb, output_pdb):
        chains = self._split_by_chain(ca_pdb)

        if not chains:
            raise ValueError(f"No se encontraron chains en {ca_pdb}")

        all_lines = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for chain_id, lines in chains.items():
                # Preprocessing
                cleaned = self._prepare_chain_for_pulchra(lines)

                if not cleaned:
                    print(f"  WARNING: is empty {chain_id} after preprocessing, skipping")
                    continue

                tmp_input  = os.path.join(tmpdir, f"chain_{chain_id}.pdb")
                tmp_output = os.path.join(tmpdir, f"chain_{chain_id}_rebuilt.pdb")

                with open(tmp_input, "w") as f:
                    f.writelines(cleaned)
                    f.write("TER\nEND\n")  # TER explícito por si acaso

                self._runPulchra(pulchra_bin, tmp_input, tmp_output)

                # Leer output y restaurar chain ID (PULCHRA lo resetea a 'A')
                with open(tmp_output) as f:
                    for line in f:
                        if line.startswith("TER") or line.startswith("END") or line.startswith("REMARK"):
                            continue
                        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                            line = line[:21] + chain_id + line[22:]
                        all_lines.append(line)

                all_lines.append("TER\n")

        with open(output_pdb, "w") as f:
            f.write("REMARK 999 REBUILT BY PULCHRA V.3.06\n")
            f.writelines(all_lines)
            f.write("END\n")


    def _runPulchra(self, pulchra_bin, ca_pdb, output_pdb):
        ca_pdb = os.path.abspath(ca_pdb)
        output_pdb = os.path.abspath(output_pdb)

        # PULCHRA escribe el output junto al input, no junto al output_pdb
        base = os.path.splitext(ca_pdb)[0]
        rebuilt = base + '.rebuilt.pdb'

        cmd = [pulchra_bin, '-c', '-q', ca_pdb]


        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f'PULCHRA failed on {ca_pdb}:\n'
                f'STDOUT: {result.stdout}\n'
                f'STDERR: {result.stderr}'
            )

        if not os.path.isfile(rebuilt):
            raise FileNotFoundError(
                f'PULCHRA output not found: {rebuilt}\n'
                f'STDOUT: {result.stdout}'
            )

        import shutil
        shutil.move(rebuilt, output_pdb)

    def createOutputStep(self):
        out_dir = self._getExtraPath('reconstructed')
        pdb_files = sorted(
            f for f in os.listdir(out_dir) if f.endswith('.pdb'))

        output_set = SetOfAtomStructs.create(self._getPath())

        for pdb_file in pdb_files:
            pdb_path = os.path.join(out_dir, pdb_file)
            atom_struct = AtomStruct()
            atom_struct.setFileName(pdb_path)
            output_set.append(atom_struct)

        self._defineOutputs(outputStructures=output_set)
        self._defineSourceRelation(self.inputEnsemble, output_set)



    def compareReconstructedRefined(self):
        """Compare RMSD between reconstructed and refined frames for backbone atoms."""
        
        out_dir     = self._getExtraPath('reconstructed')
        refined_dir = self._getExtraPath('refined')

        pdb_files = sorted(f for f in os.listdir(out_dir) if f.endswith('.pdb'))

        if not pdb_files:
            print('No frames found for comparison.')
            return

        errors = {'N': [], 'CA': [], 'C': [], 'O': []}

        for pdb_file in pdb_files:
            recon_atoms = self._parseBackbone(os.path.join(out_dir,     pdb_file))
            refin_atoms = self._parseBackbone(os.path.join(refined_dir, pdb_file))

            common = set(recon_atoms.keys()) & set(refin_atoms.keys())

            for key in common:
                _, _, atom = key
                if atom not in errors:
                    continue
                err = np.linalg.norm(
                    np.array(recon_atoms[key]) - np.array(refin_atoms[key])
                )
                errors[atom].append(err)

        print('=== Reconstructed vs Refined (backbone RMSD) ===')
        for atom, errs in errors.items():
            if errs:
                print(f'  {atom:2s}  mean: {np.mean(errs):.3f} Å  '
                    f'std: {np.std(errs):.3f} Å  '
                    f'max: {np.max(errs):.3f} Å  '
                    f'n={len(errs)}')



    # ----------- Helpers --------------
    def _convertCifToPdb(self, cif_path, pdb_path):
        from Bio.PDB import MMCIFParser, PDBIO
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('ref', cif_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)
        return pdb_path
    
    def _parseBackbone(self, pdb_path):
        """
        Parse N, CA, C, O coords from a PDB.
        Returns dict {(chain, resid, atom_name): (x, y, z)}.
        """
        atoms = {}
        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                atom = line[12:16].strip()
                if atom not in ('N', 'CA', 'C', 'O'):
                    continue
                chain = line[21]
                resid = int(line[22:26])
                coord = (float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]))
                atoms[(chain, resid, atom)] = coord
        return atoms