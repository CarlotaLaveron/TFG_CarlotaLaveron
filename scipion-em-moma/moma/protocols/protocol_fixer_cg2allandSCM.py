import os
import shutil
from collections import defaultdict

from pwem.protocols import EMProtocol

from pwchem import Plugin as pwchemPlugin
import pyworkflow.protocol.params as params
from pyworkflow import BETA
from pyworkflow.utils import Message

from pwem.objects import AtomStruct, SetOfAtomStructs
import moma

from moma.constants import CG2ALL_DIC, MOMA_DIC, PIPPACK_DIC, DIFFPACK_CG2ALL_DIC

# Constants
INPUT_ENSEMBLE = 0
INPUT_DCD      = 1

SCM_PIPPACK    = 0
SCM_DIFFPACK  = 1


class ProtocolFixerCg2allandSCM(EMProtocol):
    """
    Reconstructs backbone (N, CA, C, O) from Cα-only PDB files using cg2all
    (MainchainModel — sidechains intentionally omitted), then packs sidechains
    using a dedicated sidechain modelling tool (PIPPack).
    
    """

    _label = 'Fixer cg2all and SCM'
    _devStatus = BETA

    # Parameter definition
    def _defineParams(self, form):

        # Input section
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMode', params.EnumParam,
                      choices=['Ensemble (PDB individual por frame)',
                               'ANM MC walks (DCD)'],
                      default=INPUT_ENSEMBLE,
                      label='Input mode',
                      display=params.EnumParam.DISPLAY_HLIST,
                      important=True)

        form.addParam('inputEnsemble', params.PointerParam,
                      pointerClass='SetOfAtomStructs',
                      label='Input ensemble (Cα-only)',
                      condition='inputMode == 0',
                      important=True)

        form.addParam('inputWalk', params.PointerParam,
                      pointerClass='SetOfAtomStructs',
                      label='Input ANM MC walks',
                      condition='inputMode == 1',
                      important=True,
                      help='SetOfAtomStructs del walk — el protocolo detectará '
                           'automáticamente los DCDs asociados.')

        # Sidechain modelling section
        form.addSection(label='Sidechain Modelling')

        form.addParam('sidechainMethod', params.EnumParam,
                      choices=['PIPPack (DL, CPU)', 'DiffPack ()'],
                      default=SCM_PIPPACK,
                      label='Sidechain packing method',
                      display=params.EnumParam.DISPLAY_HLIST,
                      important=True,
                      help='PIPPack: deep learning, invariant point message passing, '
                           'best rotamer recovery (Randolph & Kuhlman, 2024).\n'
                           'DiffPack: ----.')

        # PIPPack params
        form.addParam('pippackDir', params.PathParam,
                      label='PIPPack repository path',
                      default = '/home/ubuntu/scipion/software/em/pippack-1.0.0/PIPPack/',
                      condition='sidechainMethod == 0',
                      important=True,
                      help='Absolute path to the cloned PIPPack repository '
                           '(contains ensembled_inference.py and model_weights/).\n'
                           'Clone with: git clone https://github.com/Kuhlman-Lab/PIPPack.git')

        form.addParam('numEnsembles', params.IntParam,
                      default=4,
                      label='Number of ensemble samples (PIPPack)',
                      condition='sidechainMethod == 0',
                      help='Number of independent sidechain predictions per structure. '
                           'The sample with lowest predicted RMSD is selected.\n'
                           'Recommended: 4 (good quality/speed tradeoff on CPU).\n'
                           'Range: 1 (fastest) – 8 (best quality, slowest).')

        form.addParam('pippackWeights', params.PathParam,
                      label='PIPPack model weights (optional)',
                      default='/home/ubuntu/scipion/software/em/pippack-1.0.0/PIPPack/model_weights',
                      condition='sidechainMethod == 0',
                      allowsNull=True,
                      help='Path the weights folder in the PIPPack-cloned repository. '
                           'ex:(<pippack_dir>/home/ubuntu/scipion/software/em/pippack-1.0.0/PIPPack/model_weights).')

        #DiffPack params
        form.addParam('diffpackDir', params.PathParam,
                      label='DiffPack repository path',
                      default='/home/ubuntu/scipion/software/em/diffpack-1.0.0/DiffPack',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      important=True,
                      help='Ruta absoluta al repositorio clonado de DiffPack.\n'
                           'git clone https://github.com/DeepGraphLearning/DiffPack.git\n\n'
                           'Los pesos deben descargarse manualmente desde Google Drive '
                           '(ver README del repo) y colocarse donde indique el config yaml.')

        form.addParam('diffpackConfig', params.PathParam,
                      label='DiffPack config YAML',
                      default='/home/ubuntu/scipion/software/em/diffpack-1.0.0/DiffPack/config/inference_confidence.yaml',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      important=True,
                      help='Ruta al archivo de configuración YAML.\n'
                           'Recomendado: <diffpack_dir>/config/inference_confidence.yaml\n'
                           'Alternativa: <diffpack_dir>/config/inference.yaml (sin confidence)')

        form.addParam('diffpackNumSamples', params.IntParam,
                      default=4,
                      label='Number of diffusion samples (DiffPack)',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      help='Número de muestras generadas por el proceso de difusión. '
                           'El modelo de confianza selecciona la mejor.\n'
                           'Rango: 1 (rápido) – 16 (mejor cobertura del espacio conformacional).\n'
                           'Recomendado: 8.')
        
        form.addParam('diffpackSeed', params.IntParam,
                      default=2023,
                      label='Random seed (DiffPack)',
                      condition='sidechainMethod == %d' % SCM_DIFFPACK,
                      help='Semilla aleatoria para reproducibilidad.')

    # Step insertion
    def _insertAllSteps(self):
        if self.inputMode.get() == INPUT_ENSEMBLE:
            self._insertFunctionStep(self.reconstructFromEnsembleStep, needsGPU=False)
        else:
            self._insertFunctionStep(self.reconstructFromDCDStep, needsGPU=False)

        if self.sidechainMethod.get() == SCM_PIPPACK:
            self._insertFunctionStep(self.sidechainPackingPIPPackStep, needsGPU=False)
            self._insertFunctionStep(self.fixMissingAtomsStep, needsGPU=False)
        elif self.sidechainMethod.get() == SCM_DIFFPACK:
            self._insertFunctionStep(self.sidechainPackingDiffPackStep, needsGPU=False)

        self._insertFunctionStep(self.createOutputStep, needsGPU=False)

    def fixMissingAtomsStep(self):
        pdb_dir = self._getExtraPath('allAtom_pippack')
        backup_dir = self._getExtraPath('allAtom_anomalous_backup')

        args = (
            f'--pdb_dir    "{os.path.abspath(pdb_dir)}" '
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


    # Reconstruct from PDB ensemble
    def reconstructFromEnsembleStep(self):
        ensemble = self.inputEnsemble.get()
        input_dir = self._getExtraPath('ca_input')
        output_dir = self._getExtraPath('backbone_reconstructed')
        os.makedirs(input_dir,  exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        total = ensemble.getSize()
        self.info(f'[cg2all] Total frames: {total}')

        for i, atom_struct in enumerate(ensemble.iterItems()):
            src = os.path.abspath(atom_struct.getFileName())
            dst = os.path.join(input_dir, f'frame_{i:04d}.pdb')
            shutil.copy2(src, dst)

        self._runCg2allRunner(input_dir, output_dir)

    # Reconstruct from DCD trajectory
    def reconstructFromDCDStep(self):
        ensemble = self.inputWalk.get()
        project = self.getProject().getPath()
        output_dir = self._getExtraPath('backbone_reconstructed')
        dcd_dir = self._getExtraPath('dcd_reconstructed')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(dcd_dir, exist_ok=True)

        self.info(f'[cg2all] project path : {project}')
        self.info(f'[cg2all] ensemble size : {ensemble.getSize()}')

        walk_dict = defaultdict(list)
        for atom_struct in ensemble.iterItems():
            src = atom_struct.getFileName()
            if not os.path.isabs(src):
                src = os.path.join(project, src)
            walk_dict[os.path.dirname(src)].append(src)

        self.info(f'[cg2all] Found {len(walk_dict)} walks.')

        frame_counter = 0
        for walk_dir, pdb_files in sorted(walk_dict.items()):
            pdb_files = sorted(pdb_files)
            topology  = pdb_files[0]
            dcd_files = [f for f in os.listdir(walk_dir) if f.endswith('.dcd')]
            if not dcd_files:
                self.warning(f'No DCD found in {walk_dir}, skipping.')
                continue
            dcd_path  = os.path.join(walk_dir, dcd_files[0])
            walk_name = os.path.basename(walk_dir)
            out_dcd   = os.path.join(dcd_dir, f'{walk_name}_reconstructed.dcd')
            out_top   = os.path.join(dcd_dir, f'{walk_name}_topology.pdb')

            self.info(f'[cg2all] Processing {walk_name}: {len(pdb_files)} frames')
            self._runCg2allDCD(
                topology, dcd_path, out_dcd, out_top,
                output_dir, frame_counter,
            )
            frame_counter += len(pdb_files)

        self.info(f'[cg2all] Total frames reconstructed: {frame_counter}')

    # Step 2: PIPPack sidechain packing
    def sidechainPackingPIPPackStep(self):
        backbone_dir = self._getExtraPath('backbone_reconstructed')
        allAtom_dir  = self._getExtraPath('allAtom_pippack')
        os.makedirs(allAtom_dir, exist_ok=True)

        pippack_dir = self.pippackDir.get().strip()
        num_ens     = self.numEnsembles.get()
        weights     = (self.pippackWeights.get() or '').strip()

        self.info(f'[PIPPack] backbone_dir   : {backbone_dir}')
        self.info(f'[PIPPack] allAtom_dir    : {allAtom_dir}')
        self.info(f'[PIPPack] pippack_dir    : {pippack_dir}')
        self.info(f'[PIPPack] num_ensembles  : {num_ens}')

        # Build args string for pwchemPlugin.runScript
        args = (
            f'--input_dir     "{os.path.abspath(backbone_dir)}" '
            f'--output_dir    "{os.path.abspath(allAtom_dir)}" '
            f'--pippack_dir   "{pippack_dir}" '
            f'--num_ensembles {num_ens} '
        )
        if weights:
            args += f'--model_weights "{weights}" '

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        pwchemPlugin.runScript(
            self,
            'pippack_runner.py',
            args,
            env=PIPPACK_DIC,
            cwd=self._getExtraPath(),
            scriptDir=runner_dir,
        )

        # Verify output
        out_pdbs = [f for f in os.listdir(allAtom_dir) if f.endswith('.pdb')]
        self.info(f'[PIPPack] {len(out_pdbs)} all-atom PDB files generated.')
        if not out_pdbs:
            raise RuntimeError(
                'PIPPack produced no output PDBs. '
                'Check the log above for errors.'
            )

    def sidechainPackingDiffPackStep(self):
        backbone_dir  = self._getExtraPath('backbone_reconstructed')
        allAtom_dir   = self._getExtraPath('allAtom_diffpack')
        os.makedirs(allAtom_dir, exist_ok=True)

        diffpack_dir  = self.diffpackDir.get().strip()
        config_path   = self.diffpackConfig.get().strip()
        num_samples   = self.diffpackNumSamples.get()
        seed          = self.diffpackSeed.get()

        self.info(f'[DiffPack] backbone_dir  : {backbone_dir}')
        self.info(f'[DiffPack] allAtom_dir   : {allAtom_dir}')
        self.info(f'[DiffPack] diffpack_dir  : {diffpack_dir}')
        self.info(f'[DiffPack] config        : {config_path}')
        self.info(f'[DiffPack] num_samples   : {num_samples}')
        self.info(f'[DiffPack] seed          : {seed}')

        args = (
            f'--input_dir    "{os.path.abspath(backbone_dir)}" '
            f'--output_dir   "{os.path.abspath(allAtom_dir)}" '
            f'--diffpack_dir "{diffpack_dir}" '
            f'--config       "{config_path}" '
            f'--num_samples  {num_samples} '
            f'--seed         {seed} '
        )

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        pwchemPlugin.runScript(
            self,
            'diffpack_runner.py',
            args,
            env=DIFFPACK_CG2ALL_DIC,
            cwd=self._getExtraPath(),
            scriptDir=runner_dir,
        )

        out_pdbs = [f for f in os.listdir(allAtom_dir) if f.endswith('.pdb')]
        self.info(f'[DiffPack] {len(out_pdbs)} all-atom PDB files generated.')
        if not out_pdbs:
            raise RuntimeError(
                'DiffPack produced no output PDBs. '
                'Check the log above for errors.'
            )


    # ── Output step ───────────────────────────────────────────────────────────
    def createOutputStep(self):
        method = self.sidechainMethod.get()

        if method == SCM_PIPPACK:
            candidate_dir = self._getExtraPath('allAtom_pippack')
            label = 'PIPPack'
        elif method == SCM_DIFFPACK:
            candidate_dir = self._getExtraPath('allAtom_diffpack')
            label = 'DiffPack'
        else:
            candidate_dir = None
            label = None

        backbone_dir = self._getExtraPath('backbone_reconstructed')

        if (candidate_dir
                and os.path.isdir(candidate_dir)
                and any(f.endswith('.pdb') for f in os.listdir(candidate_dir))):
            output_dir = candidate_dir
            self.info(f'[Output] Using all-atom {label} structures.')
        else:
            output_dir = backbone_dir
            self.info('[Output] Using backbone-only structures (SCM not run or failed).')

        pdb_files  = sorted(f for f in os.listdir(output_dir) if f.endswith('.pdb'))
        output_set = SetOfAtomStructs.create(self._getPath())

        for pdb_file in pdb_files:
            pdb_path    = os.path.join(output_dir, pdb_file)
            atom_struct = AtomStruct()
            atom_struct.setFileName(pdb_path)
            output_set.append(atom_struct)

        self._defineOutputs(outputStructures=output_set)

        if self.inputMode.get() == INPUT_ENSEMBLE:
            self._defineSourceRelation(self.inputEnsemble, output_set)
        else:
            self._defineSourceRelation(self.inputWalk, output_set)

        self.info(f'[Output] {len(pdb_files)} all-atom structures registered.')


    # Helpers
    def _runCg2allDCD(self, topology, dcd_path, out_dcd, out_top,
                      output_dir, frame_start):
        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        args = (
            f'--topology    "{topology}" '
            f'--dcd         "{dcd_path}" '
            f'--out_dcd     "{os.path.abspath(out_dcd)}" '
            f'--out_top     "{os.path.abspath(out_top)}" '
            f'--output_dir  "{os.path.abspath(output_dir)}" '
            f'--frame_start {frame_start}'
        )

        pwchemPlugin.runScript(
            self,
            'cg2all_runner_DCD.py',
            args,
            env=CG2ALL_DIC,
            cwd=self._getExtraPath(),
            scriptDir=runner_dir,
        )

    def _runCg2allRunner(self, input_dir, output_dir):
        import glob
        import mdtraj as md

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        pdb_files = sorted(glob.glob(os.path.join(input_dir, 'frame_*.pdb')))
        if not pdb_files:
            raise FileNotFoundError(f'No frame PDBs found in {input_dir}')

        topology = pdb_files[0]
        traj     = md.load(pdb_files, top=topology)
        dcd_path = os.path.join(input_dir, 'ensemble.dcd')
        traj.save_dcd(dcd_path)
        self.info(f'[cg2all] Converted {len(pdb_files)} PDBs to DCD: {dcd_path}')

        out_dcd = os.path.join(output_dir, 'ensemble_reconstructed.dcd')
        out_top = os.path.join(output_dir, 'ensemble_topology.pdb')

        args = (
            f'--topology    "{os.path.abspath(topology)}" '
            f'--dcd         "{os.path.abspath(dcd_path)}" '
            f'--out_dcd     "{os.path.abspath(out_dcd)}" '
            f'--out_top     "{os.path.abspath(out_top)}" '
            f'--output_dir  "{os.path.abspath(output_dir)}" '
            f'--frame_start 0'
        )

        pwchemPlugin.runScript(
            self,
            'cg2all_BB_runner.py',
            args,
            env=CG2ALL_DIC,
            cwd=self._getExtraPath(),
            scriptDir=runner_dir,
        )