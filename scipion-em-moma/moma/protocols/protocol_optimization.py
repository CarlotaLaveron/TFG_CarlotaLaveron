"""
Geometry optimization of protein centroids using PDBFixer + OpenMM.

Strategy: 3-phase progressive minimization with AMBER14 + GBn2 implicit solvent.
  Phase 1: hydrogens only        (heavy atoms fixed)
  Phase 2: sidechains            (backbone fixed: N, Cα, C, O, OXT)
  Phase 3: global                (Cα FIXED, N/C/O soft restraint)

No experimental map required.
"""

import os
import moma

from pyworkflow.constants import BETA
import pyworkflow.protocol.params as params
from pyworkflow.utils import Message
from pwem.protocols import EMProtocol
from pwem.objects import SetOfAtomStructs, AtomStruct
from pwchem import Plugin as pwchemPlugin
from moma.constants import MOMA_DIC

class ProtocolOptimization(EMProtocol):
    """
    Geometry optimization using PDBFixer + OpenMM with Cα position restraints.
    Input:  SetOfAtomStructs (e.g. centroids from WARIO).
    Output: SetOfAtomStructs with optimized geometries (backbone preserved).
    """
 
    _label = 'Geometry Optimization'
    _devStatus = BETA
 
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
 
        form.addParam(
            'inputStructures',
            params.PointerParam,
            pointerClass='SetOfAtomStructs',
            label='Input structures',
            important=True,
            help='SetOfAtomStructs to optimize (e.g. centroids from WARIO).'
        )
 
        form.addSection(label='Optimization settings')
 
        form.addParam(
            'maxIterations',
            params.IntParam,
            default=150,
            label='Max minimization iterations',
            help='Maximum OpenMM energy minimization steps. '
                 '100 is sufficient to resolve clashes on CPU. '
                 '0 = minimize until convergence (very slow).'
        )
 
        form.addParam(
            'caForceConstant',
            params.FloatParam,
            default=200.0,
            label='C_alfa restraint force constant (kcal/mol/Å²)',
            help='Force constant for C_alfa position restraints. '
                 'Higher values keep the backbone closer to the original. '
                 'Recommended: 100.0 (strong) to 10.0 (soft). '
                 'Set to 0 to disable restraints (not recommended).'
        )
 
    def _insertAllSteps(self):
        self._insertFunctionStep(self.optimizeStructuresStep, needsGPU=False)
        self._insertFunctionStep(self.createOutputStep, needsGPU=False)
 
    def optimizeStructuresStep(self):
        extra_path = os.path.abspath(self._getExtraPath())
        runner_dir = os.path.join(
            os.path.dirname(os.path.abspath(moma.__file__)),
            'protocols', 'scripts'
        )
 
        pdb_paths = []
        for atom_struct in self.inputStructures.get():
            pdb_paths.append(os.path.abspath(atom_struct.getFileName()))
        pdb_list = ','.join(pdb_paths)
 
        args = (
            ' --pdb_list "{}"'
            ' --output_dir "{}"'
            ' --max_iterations {}'
            ' --ca_force_constant {}'
        ).format(pdb_list, extra_path,
                 self.maxIterations.get(),
                 self.caForceConstant.get())
 
        pwchemPlugin.runScript(
            self,
            'optimization_runner.py',
            args,
            env=MOMA_DIC,
            cwd=extra_path,
            scriptDir=runner_dir,
        )
 
    def createOutputStep(self):
        extra_path = self._getExtraPath()
        output_set = SetOfAtomStructs.create(self._getPath())
 
        for pdb_file in sorted(
            f for f in os.listdir(extra_path) if f.endswith('_optimized.pdb')
        ):
            atom_struct = AtomStruct()
            atom_struct.setFileName(os.path.join(extra_path, pdb_file))
            output_set.append(atom_struct)
 
        self._defineOutputs(outputOptimized=output_set)
        self._defineSourceRelation(self.inputStructures, output_set)
        self.info(f'Done: {output_set.getSize()} optimized structures.')
 
    def _summary(self):
        summary = []
        if self.inputStructures.get():
            summary.append(f'Input: {self.inputStructures.get().getSize()} structures')
        summary.append(f'Max iterations: {self.maxIterations.get()}')
        summary.append(f'Cα force constant: {self.caForceConstant.get()} kcal/mol/Å²')
        return summary
