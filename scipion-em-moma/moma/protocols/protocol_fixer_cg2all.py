# ============================================================
#  moma/protocols/protocol_fixer_cg2all.py
# ============================================================

from collections import defaultdict
import os
import shutil

from pwem.protocols import EMProtocol
from pwem.objects import AtomStruct, SetOfAtomStructs
from pyworkflow import BETA
from pyworkflow.protocol import params
from pyworkflow.utils import Message

import moma
from moma.constants import CG2ALL_DIC
from pwchem import Plugin as pwchemPlugin

INPUT_ENSEMBLE = 0
INPUT_DCD      = 1


class ProtocolFixerCg2all(EMProtocol):
    """
    Reconstructs full backbone (N, CA, C, O, CB + side chains) from
    Cα-only PDB files using cg2all (deep learning, CalphaBasedModel).

    cg2all preserves the original Cα coordinates (--fix) and predicts
    all remaining atoms using an SE3-equivariant neural network.

    Reference:
        Heo & Feig, bioRxiv (2023).
        https://github.com/huhlim/cg2all
    """
    _label = 'Fixer cg2all'
    _devStatus = BETA


    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMode', params.EnumParam,
              choices=['Ensemble (PDB individual por frame)', 'ANM MC walks (DCD)'],
              default=0,
              label='Input mode',
              display=params.EnumParam.DISPLAY_HLIST,
              important=True)

        form.addParam('inputEnsemble', params.PointerParam,
                    pointerClass='SetOfAtomStructs',
                    label='Input ensemble (C_alpha-only)',
                    condition='inputMode == 0',
                    important=True)

        form.addParam('inputWalk', params.PointerParam,
                    pointerClass='SetOfAtomStructs',
                    label='Input ANM MC walks',
                    condition='inputMode == 1',
                    important=True,
                    help='SetOfAtomStructs del walk — el protocolo detectará '
                        'automáticamente los DCDs asociados.')

    def _insertAllSteps(self):
        if self.inputMode.get() == 0:
            self._insertFunctionStep(self.reconstructFromEnsembleStep, needsGPU=False)
        else:
            self._insertFunctionStep(self.reconstructFromDCDStep, needsGPU=False)

        self._insertFunctionStep(self.createOutputStep, needsGPU=False)


    def reconstructFromEnsembleStep(self):
        ensemble = self.inputEnsemble.get()
        input_dir = self._getExtraPath('ca_input')
        output_dir = self._getExtraPath('reconstructed')
        os.makedirs(input_dir,  exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        total = ensemble.getSize()
        self.info(f'[DEBUG] Total frames: {total}')

        for i, atom_struct in enumerate(ensemble.iterItems()):
            src = os.path.abspath(atom_struct.getFileName())
            dst = os.path.join(input_dir, f'frame_{i:04d}.pdb')
            shutil.copy2(src, dst)
        self._runCg2allRunner(input_dir, output_dir)

    def reconstructFromDCDStep(self):
        ensemble = self.inputWalk.get()
        project = self.getProject().getPath()
        output_dir = self._getExtraPath('reconstructed')
        dcd_dir = self._getExtraPath('dcd_reconstructed')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(dcd_dir,    exist_ok=True)

        #self.info(f'[DEBUG] project path: {project}')
        #self.info(f'[DEBUG] ensemble size: {ensemble.getSize()}')

        
        walk_dict = defaultdict(list)

        for atom_struct in ensemble.iterItems():
            src = atom_struct.getFileName()
            if not os.path.isabs(src):
                src = os.path.join(project, src)
            walk_dict[os.path.dirname(src)].append(src)
 
        #self.info(f'[cg2all] Found {len(walk_dict)} walks.')
 
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

            #self.info(f'[cg2all] Processing {walk_name}: {len(pdb_files)} frames')
            self._runCg2allDCD(topology, dcd_path, out_dcd, out_top, output_dir, frame_counter)
            frame_counter += len(pdb_files)
 
        #self.info(f'[cg2all] Total frames reconstructed: {frame_counter}')
 
    def _runCg2allDCD(self, topology, dcd_path, out_dcd, out_top, output_dir, frame_start):
        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        args = (
            f'--topology "{topology}" '
            f'--dcd "{dcd_path}" '
            f'--out_dcd "{os.path.abspath(out_dcd)}" '
            f'--out_top "{os.path.abspath(out_top)}" '
            f'--output_dir "{os.path.abspath(output_dir)}" '
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
        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')

        args = (
            f'--input_dir "{os.path.abspath(input_dir)}" '
            f'--output_dir "{os.path.abspath(output_dir)}"'
        )

        pwchemPlugin.runScript(
            self,
            'cg2all_runner.py',
            args,
            env=CG2ALL_DIC,
            cwd=output_dir,
            scriptDir=runner_dir,
        )


    def createOutputStep(self):
        output_dir = self._getExtraPath('reconstructed')
        pdb_files  = sorted(
            f for f in os.listdir(output_dir) if f.endswith('.pdb')
        )

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