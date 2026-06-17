import os
import shutil

from pwem.protocols import EMProtocol
from pwchem import Plugin as pwchemPlugin
import pyworkflow.protocol.params as params
from pyworkflow import BETA
from pyworkflow.utils import Message
from pwem.objects import AtomStruct, SetOfAtomStructs
import moma
import copy
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBIO
from Bio.PDB import MMCIFParser, Select
from collections import Counter
from Bio.PDB import PDBParser

from moma.constants import PIPPACK_DIC, DIFFPACK_IDW_DIC, MOMA_DIC
from moma.objects import InvDistTree3D

# ── Constants ─────────────────────────────────────────────────────────────────
SCM_PIPPACK  = 0
SCM_DIFFPACK = 1


class ProtocolIDWandSCM(EMProtocol):
    """
    Reconstructs full-atom protein structures from a Cα-only ensemble using
    Inverse Distance Weighting (IDW), then strips sidechains and repacks them
    using a dedicated sidechain modelling tool (PIPPack or DiffPack).

    Pipeline:
        Cα-only ensemble → IDW (full-atom) → strip sidechains → SCM → output
    """

    _label    = 'IDW and SCM'
    _devStatus = BETA


    def _defineParams(self, form):

        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputReference', params.PointerParam,
                      pointerClass='AtomStruct',
                      label='Reference full-atom structure',
                      important=True,
                      help='PDB/CIF file containing ALL atoms. Its C_alfa positions '
                           'serve as source control points for the IDW interpolation.')

        form.addParam('inputEnsemble', params.PointerParam,
                      pointerClass='SetOfAtomStructs',
                      label='Target C_alfa ensemble',
                      important=True,
                      help='Set of C_alfa-only models (one per conformer).')

        form.addSection(label='IDW parameters')

        form.addParam('searchRadius', params.FloatParam,
                      default=15.0,
                      label='Search radius R (Å)',
                      help='Maximum distance in Angstroms to search for C_alfa neighbours.\n'
                           'Typical values: 10-20 Å.\n'
                           'Formula: w_k = ((R - d) / (R * d)) ^ p')

        
        form.addSection(label='Sidechain Modelling')

        form.addParam('sidechainMethod', params.EnumParam,
                      choices=['PIPPack (DL, CPU)', 'DiffPack (DL, CPU)'],
                      default=SCM_PIPPACK,
                      label='Sidechain packing method',
                      display=params.EnumParam.DISPLAY_HLIST,
                      important=True,
                      help='PIPPack: invariant point message passing, best rotamer '
                           'recovery (Randolph & Kuhlman, 2024).\n'
                           'DiffPack: torsional diffusion model (Zhang et al., NeurIPS 2023).')

        # PIPPack params
        form.addParam('pippackDir', params.PathParam,
                      label='PIPPack repository path',
                      default='/home/ubuntu/scipion/software/em/pippack-1.0.0/PIPPack/',
                      condition='sidechainMethod == %d' % SCM_PIPPACK,
                      important=True,
                      help='Absolute path to the cloned PIPPack repository.')

        form.addParam('numEnsembles', params.IntParam,
                      default=4,
                      label='Number of ensemble samples',
                      condition='sidechainMethod == %d' % SCM_PIPPACK,
                      help='Number of independent sidechain predictions per structure.\n'
                           'Range: 1 (fastest) – 8 (best quality).')

        form.addParam('pippackWeights', params.PathParam,
                      label='PIPPack model weights',
                      default='/home/ubuntu/scipion/software/em/pippack-1.0.0/PIPPack/model_weights',
                      condition='sidechainMethod == %d' % SCM_PIPPACK,
                      allowsNull=True,
                      help='Path to the weights folder in the PIPPack repository.')

        # DiffPack params
        form.addParam('diffpackDir', params.PathParam,
                      label='DiffPack repository path',
                      default='/home/ubuntu/scipion/software/em/diffpack-1.0.0/DiffPack',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      important=True,
                      help='Absolute path to the cloned DiffPack repository.')

        form.addParam('diffpackConfig', params.PathParam,
                      label='DiffPack config YAML',
                      default='/home/ubuntu/scipion/software/em/diffpack-1.0.0/DiffPack/config/inference_confidence.yaml',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      important=True,
                      help='Path to DiffPack inference YAML config.')

        form.addParam('diffpackNumSamples', params.IntParam,
                      default=4,
                      label='Number of diffusion samples',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      help='Number of diffusion samples per structure.\n'
                           'Range: 1 (fast) – 16 (best coverage).')

        form.addParam('diffpackSeed', params.IntParam,
                      default=2023,
                      label='Random seed',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      help='Random seed for reproducibility.')

    def _insertAllSteps(self):
        self._insertFunctionStep(self.filterCalphaStep, needsGPU=False)
        self._insertFunctionStep(self.idwReconstructionStep,  needsGPU=False)

        if self.sidechainMethod.get() == SCM_PIPPACK:
            self._insertFunctionStep(self.stripSidechainsStep, needsGPU=False)
            self._insertFunctionStep(self.sidechainPackingPIPPackStep, needsGPU=False)
            self._insertFunctionStep(self.fixMissingAtomsStep, needsGPU=False)

        elif self.sidechainMethod.get() == SCM_DIFFPACK:
            self._insertFunctionStep(self.sidechainPackingDiffPackStep, needsGPU=False)

        self._insertFunctionStep(self.createOutputStep, needsGPU=False)

    def _parse_structure(self, path, struct_id):
        from Bio.PDB import PDBParser, MMCIFParser
        ext = os.path.splitext(path)[1].lower()
        parser = MMCIFParser(QUIET=True) if ext in ('.cif', '.mmcif') \
                else PDBParser(QUIET=True)
        return parser.get_structure(struct_id, path)


    def fixMissingAtomsStep(self):
        pdb_dir = self._getExtraPath('allAtom_pippack')
        backup_dir = self._getExtraPath('allAtom_anomalous_backup')

        args = (
            f'--pdb_dir "{os.path.abspath(pdb_dir)}" '
            f'--backup_dir "{os.path.abspath(backup_dir)}" '
        )

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        pwchemPlugin.runScript(
            self,
            'fix_atoms_runner.py',
            args,
            env=MOMA_DIC,
            cwd=self._getExtraPath(),
            scriptDir=runner_dir,
        )
        
    def idwReconstructionStep(self):

        R = self.searchRadius.get()
        ref_path = self.inputReference.get().getFileName()
        ensemble = self.inputEnsemble.get()
        out_dir = self._getExtraPath('idw_reconstructed')
        os.makedirs(out_dir, exist_ok=True)

        ref_struct = self._parse_structure(ref_path, 'reference')

        src_ca_coords, src_ca_keys = self._extract_ca_coords(ref_struct)
        all_atom_coords, all_atoms = self._extract_all_atom_coords(ref_struct)

        if src_ca_coords.shape[0] == 0:
            raise RuntimeError(
                'No Cα atoms found in reference structure: %s' % ref_path
            )

        # k automático como el primero
        tree_ref = cKDTree(src_ca_coords)
        counts = [len(tree_ref.query_ball_point(ca, r=R)) for ca in src_ca_coords]
        k = max(int(np.mean(counts) * 2), 1)

        self.info(f'[IDW] Reference: {len(src_ca_keys)} Cα, {len(all_atoms)} total atoms')
        self.info(f'[IDW] R={R:.1f} Å, k={k} (auto), p=2.0')

        io = PDBIO()
        idw = InvDistTree3D(src_ca_coords, leafsize=10)


        calpha_dir = self._getExtraPath('calpha_filtered')
        
        for i, target_as in enumerate(ensemble):
            target_path = os.path.join(calpha_dir, f'calpha_{i:04d}.pdb')
            target_struct = self._parse_structure(target_path, f'target_{i}')
            dst_ca_coords, dst_ca_keys = self._extract_ca_coords(target_struct)
            ref_struct = self._parse_structure(ref_path, 'reference')
            all_atom_coords, all_atoms = self._extract_all_atom_coords(ref_struct)

            src_aligned, dst_aligned = self._align_ca_by_key(
                src_ca_coords, src_ca_keys,
                dst_ca_coords, dst_ca_keys,
            )

            if src_ca_coords.shape != dst_ca_coords.shape:
                raise ValueError(
                    "src_ca_coords and dst_ca_coords must have the same shape, "
                    "got %s vs %s" % (src_ca_coords.shape, dst_ca_coords.shape)
                )

            if src_aligned.shape[0] == 0:
                self.warning(f'[IDW] No matching Cα for model {i} — skipping.')
                continue

            #self.info(f'[IDW] [{i+1}/{len(ensemble)}] {src_aligned.shape[0]} matched Cα pairs')

            
            displacements = dst_aligned - src_aligned
            interpolated  = idw(all_atom_coords, displacements, R=R, k=k)
            new_coords    = all_atom_coords + interpolated

            out_struct = copy.deepcopy(ref_struct)
            _, out_atoms = self._extract_all_atom_coords(out_struct)
            self._apply_coords(out_atoms, new_coords)

            out_path = os.path.join(out_dir, f'frame_{i:04d}.pdb')
            io.set_structure(out_struct)
            io.save(out_path)
            self._fix_pdb_format(out_path)

            self.info(f'[IDW] Frame {i+1} done: {out_path}')

        #self.info('[IDW] Reconstruction complete.')

    def stripSidechainsStep(self):
        idw_dir = self._getExtraPath('idw_reconstructed')
        backbone_dir = self._getExtraPath('backbone_reconstructed')
        os.makedirs(backbone_dir, exist_ok=True)

        backbone_atoms = {'N', 'CA', 'C', 'O', 'CB'}
        pdb_files = sorted(f for f in os.listdir(idw_dir) if f.endswith('.pdb'))

        for pdb_file in pdb_files:
            in_path = os.path.join(idw_dir,      pdb_file)
            out_path = os.path.join(backbone_dir, pdb_file)

            atom_counter = 1
            with open(in_path) as fh_in, open(out_path, 'w') as fh_out:
                for line in fh_in:
                    record = line[:6].strip()
                    if record in ('MODEL', 'ENDMDL'):
                        continue
                    if record in ('ATOM', 'HETATM'):
                        atom_name = line[12:16].strip()
                        if atom_name not in backbone_atoms:
                            continue
                        line = line[:6] + f'{atom_counter:5d}' + line[11:]
                        atom_counter += 1
                    fh_out.write(line)

        #self.info(f'[Strip] {len(pdb_files)} backbone-only PDBs written to {backbone_dir}')

    def sidechainPackingPIPPackStep(self):
        backbone_dir = self._getExtraPath('backbone_reconstructed')
        allAtom_dir = self._getExtraPath('allAtom_pippack')
        os.makedirs(allAtom_dir, exist_ok=True)

        pippack_dir = self.pippackDir.get().strip()
        num_ens = self.numEnsembles.get()
        weights = (self.pippackWeights.get() or '').strip()

        #self.info(f'[PIPPack] backbone_dir  : {backbone_dir}')
        #self.info(f'[PIPPack] allAtom_dir   : {allAtom_dir}')
        #self.info(f'[PIPPack] pippack_dir   : {pippack_dir}')
        #self.info(f'[PIPPack] num_ensembles : {num_ens}')

        args = (
            f'--input_dir "{os.path.abspath(backbone_dir)}" '
            f'--output_dir "{os.path.abspath(allAtom_dir)}" '
            f'--pippack_dir "{pippack_dir}" '
            f'--num_ensembles {num_ens} '
        )
        if weights:
            args += f'--model_weights "{weights}" '

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        pwchemPlugin.runScript(
            self, 'pippack_runner.py', args,
            env=PIPPACK_DIC, cwd=self._getExtraPath(), scriptDir=runner_dir,
        )

        out_pdbs = [f for f in os.listdir(allAtom_dir) if f.endswith('.pdb')]
        self.info(f'[PIPPack] {len(out_pdbs)} all-atom PDB files generated.')
        if not out_pdbs:
            raise RuntimeError('PIPPack produced no output PDBs. Check the log above.')

    def sidechainPackingDiffPackStep(self):
        #self.info(f'[DiffPack DEBUG] extra path: {self._getExtraPath()}')
        idw_dir = os.path.abspath(self._getExtraPath('idw_reconstructed'))
        input_dir = os.path.abspath(self._getExtraPath('diffpack_input'))
        allAtom_dir = os.path.abspath(self._getExtraPath('allAtom_diffpack'))
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(allAtom_dir, exist_ok=True)

        for pdb_file in sorted(f for f in os.listdir(idw_dir) if f.endswith('.pdb')):
            in_path = os.path.join(idw_dir,   pdb_file)
            out_path = os.path.join(input_dir, pdb_file)
            with open(in_path) as fh_in, open(out_path, 'w') as fh_out:
                for line in fh_in:
                    if line.startswith('HETATM'):
                        continue
                    fh_out.write(line)

        diffpack_dir = self.diffpackDir.get().strip()
        config_path = self.diffpackConfig.get().strip()
        num_samples = self.diffpackNumSamples.get()
        seed = self.diffpackSeed.get()

        #self.info(f'[DiffPack] input_dir : {input_dir}')
        #self.info(f'[DiffPack] allAtom_dir : {allAtom_dir}')
        #self.info(f'[DiffPack] diffpack_dir: {diffpack_dir}')
        #self.info(f'[DiffPack] config : {config_path}')
        #self.info(f'[DiffPack] num_samples : {num_samples}')
        #self.info(f'[DiffPack] seed : {seed}')

        args = (
            f'--input_dir "{os.path.abspath(input_dir)}" '
            f'--output_dir "{os.path.abspath(allAtom_dir)}" '
            f'--diffpack_dir "{diffpack_dir}" '
            f'--config "{config_path}" '
            f'--num_samples {num_samples} '
            f'--seed {seed} '
        )

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        #self.info(f'[DiffPack DEBUG] args: {args}')

        pwchemPlugin.runScript(
            self, 'diffpack_runner.py', args,
            env=DIFFPACK_IDW_DIC, cwd=self._getExtraPath(), scriptDir=runner_dir,
        )

        out_pdbs = [f for f in os.listdir(allAtom_dir) if f.endswith('.pdb')]
        #self.info(f'[DiffPack] {len(out_pdbs)} all-atom PDB files generated.')
        if not out_pdbs:
            raise RuntimeError('DiffPack produced no output PDBs. Check the log above.')

    def createOutputStep(self):
        method = self.sidechainMethod.get()

        if method == SCM_PIPPACK:
            candidate_dir = self._getExtraPath('allAtom_pippack')
            label = 'PIPPack'
            backbone_dir = self._getExtraPath('backbone_reconstructed')
        else:
            candidate_dir = self._getExtraPath('allAtom_diffpack')
            label = 'DiffPack'
            backbone_dir = self._getExtraPath('idw_reconstructed')

        

        if (os.path.isdir(candidate_dir)
                and any(f.endswith('.pdb') for f in os.listdir(candidate_dir))):
            output_dir = candidate_dir
            #self.info(f'[Output] Using all-atom {label} structures.')
        else:
            output_dir = backbone_dir
            #self.info('[Output] Using backbone-only structures (SCM not run or failed).')

        pdb_files = sorted(f for f in os.listdir(output_dir) if f.endswith('.pdb'))
        output_set = SetOfAtomStructs.create(self._getPath())

        for pdb_file in pdb_files:
            pdb_path = os.path.join(output_dir, pdb_file)
            atom_struct = AtomStruct()
            atom_struct.setFileName(pdb_path)
            output_set.append(atom_struct)

        self._defineOutputs(outputStructures=output_set)
        self._defineSourceRelation(self.inputReference, output_set)
        self._defineSourceRelation(self.inputEnsemble,  output_set)
        #self.info(f'[Output] {len(pdb_files)} all-atom structures registered.')

    def filterCalphaStep(self):

        class CalphaSelect(Select):
            def accept_atom(self, atom):
                return atom.get_name() == 'CA'

        ensemble = self.inputEnsemble.get()
        out_dir  = self._getExtraPath('calpha_filtered')
        os.makedirs(out_dir, exist_ok=True)

        for i, target_as in enumerate(ensemble):
            target_path = target_as.getFileName()
            ext = os.path.splitext(target_path)[1].lower()
            parser = MMCIFParser(QUIET=True) if ext in ('.cif', '.mmcif') else PDBParser(QUIET=True)
            struct = parser.get_structure(f'target_{i}', target_path)

            out_path = os.path.join(out_dir, f'calpha_{i:04d}.pdb')
            io = PDBIO()
            io.set_structure(struct)
            io.save(out_path, CalphaSelect())

        #self.info(f'[FilterCα] {len(ensemble)} structures filtered to Cα only in {out_dir}')


    def _extract_ca_coords(self, structure):
        import numpy as np
        coords, keys = [], []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    if 'CA' not in residue:
                        continue
                    coords.append(residue['CA'].get_vector().get_array())
                    keys.append((chain.id, residue.id[1], residue.id[2]))
            break
        return np.array(coords, dtype=np.float64), keys

    def _extract_all_atom_coords(self, structure):
        import numpy as np
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

    @staticmethod
    def _align_ca_by_key(src_coords, src_keys, dst_coords, dst_keys):
        import numpy as np
        dst_map = dict(zip(dst_keys, dst_coords))
        src_aligned, dst_aligned = [], []
        for key, coord in zip(src_keys, src_coords):
            if key in dst_map:
                src_aligned.append(coord)
                dst_aligned.append(dst_map[key])
        if not src_aligned:
            return np.empty((0, 3)), np.empty((0, 3))
        return np.array(src_aligned), np.array(dst_aligned)

    @staticmethod
    def _apply_coords(atoms, new_coords):
        import numpy as np
        for atom, coord in zip(atoms, new_coords):
            atom.set_coord(coord)

    @staticmethod
    def _fix_pdb_format(pdb_path):
        with open(pdb_path) as f:
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
                        atom_count, res_name, chain, resseq))
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
                atom_count, res_name, chain, resseq))
        fixed.append('END\n')
        with open(pdb_path, 'w') as f:
            f.writelines(fixed)

    def _validate(self):
        errors = []
        if self.inputReference.get() is None:
            errors.append('A reference full-atom structure must be provided.')
        ens = self.inputEnsemble.get()
        if ens is None:
            errors.append('A target Cα ensemble must be provided.')
        elif len(ens) == 0:
            errors.append('The target ensemble is empty.')
        if self.searchRadius.get() <= 0:
            errors.append('Search radius R must be positive.')
        return errors

    def _summary(self):
        out = getattr(self, 'outputStructures', None)
        if self.isFinished() and out is not None:
            return [
                f'Reconstructed {len(out)} full-atom structure(s).',
                f'IDW R={self.searchRadius.get():.1f} Å, p=2.0.',
                f'SCM: {"PIPPack" if self.sidechainMethod.get() == SCM_PIPPACK else "DiffPack"}.',
            ]
        return ['Protocol not finished yet.']