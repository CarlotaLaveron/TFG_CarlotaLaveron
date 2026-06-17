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
This protocol runs WARIO (Weighted contacts And Residue Interaction analysis
Of ensembles) to analyse the conformational heterogeneity of a protein ensemble.

Given a protein topology file (.pdb) and a trajectory file (.xtc, or a folder
of .pdb files), WARIO:
  1. Computes weighted residue-residue contact matrices for every conformation.
  2. Embeds the contact data in a low-dimensional UMAP space.
  3. Clusters the conformations with HDBSCAN.
  4. Identifies the medoid (most representative frame) of each cluster. 

References:
  WARIO: https://gitlab.laas.fr/moma/methods/analysis/WARIO

"""

from enum import Enum

from moma.constants import MOMA_DIC
from pyworkflow.constants import BETA
import pyworkflow.protocol.params as params
from pyworkflow.utils import Message
from pwchem.objects import MDSystem
from pwem.objects import SetOfAtomStructs, AtomStruct
from pwem.protocols import EMProtocol
from pwchem.constants import OPENBABEL_DIC
from pwchem import Plugin as pwchemPlugin
import os
import shutil
import moma
import numpy as np

from pwchem import Plugin as pwchemPlugin

INPUT_PATHS = 0
INPUT_ENSEMBLE = 1
INPUT_MULTIFRAME = 2

class ProtocolMomaWario(EMProtocol):
    """
    This protocol will print hello world in the console
    IMPORTANT: Classes names should be unique, better prefix them
    """
    _label = 'Wario'
    _devStatus = BETA
    def _defineParams(self, form):

        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMode', params.EnumParam,
                        choices=['File paths (.pdb & .xtc)', 'Ensemble of proteins', 'Multiframe .pdb'],
                        default=INPUT_PATHS,
                        label='Input mode',
                        important=True,
                        display=params.EnumParam.DISPLAY_HLIST,
                        help='Choose whether to provide raw file paths or a Scipion ensemble object.')

        
        form.addParam('inputAtomStruct', params.PointerParam,
              pointerClass='AtomStruct', allowsNull=False,
              label="Input atom structure",
              condition='inputMode == %d' % INPUT_PATHS, 
              help='Select the atom structure.')

        form.addParam('inputTrajectory', params.PointerParam,
                    pointerClass='SetOfTrajFrames',
                    allowsNull=False,
                    label="Input trajectory",
                    condition='inputMode == %d' % INPUT_PATHS,  
                    help='Select the MD trajectory.')

        form.addParam('inputEnsemble', params.PointerParam,
                    pointerClass='SetOfAtomStructs', 
                    label='Protein ensemble',
                    condition='inputMode == %d' % INPUT_ENSEMBLE,
                    important=True,
                    help='Select a Scipion ensemble object (SetOfAtomStructs).')
        
        form.addParam('inputMultiframePdb', params.PathParam,
                      label='Multiframe .pdb file',
                      condition='inputMode == %d' % INPUT_MULTIFRAME,
                      help='Path to a single multiframe .pdb file.')

        form.addParam('useSubsequence', params.BooleanParam,
              default=False,
              label='Use subsequence?',
              help='Select Yes to analyse only a subset of residues.')

        form.addParam('subsequence', params.StringParam,
                    label='Residue ranges',
                    condition='useSubsequence',
                    help='Residue ranges to analyse, separated by commas. '
                        'Example: "100-150, 170-180" will analyse residues '
                        '100 to 150 and 170 to 180.')

    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInput, needsGPU=False)   
        self._insertFunctionStep(self.callWario, needsGPU=False)
        self._insertFunctionStep(self.createOutputStep, needsGPU=False)

    def convertInput(self):
        mode = self.inputMode.get()
        if mode == INPUT_PATHS:
            self.convertFromPaths()
        elif mode == INPUT_ENSEMBLE:
            self.convertFromEnsemble()
        elif mode == INPUT_MULTIFRAME:
            self.convertFromMultiframe()

    def convertFromPaths(self):
        structure = self.inputAtomStruct.get().getFileName()  # corregido
        trajectory = self.inputTrajectory.get().getFileName()  # corregido

        structure_name_file = os.path.basename(structure)
        trajectory_name_file = os.path.basename(trajectory)

        structure_dest = self._getTmpPath(structure_name_file)
        trajectory_dest = self._getTmpPath(trajectory_name_file)

        if not os.path.exists(structure_dest):
            os.symlink(os.path.abspath(structure), structure_dest)
        if not os.path.exists(trajectory_dest):
            os.symlink(os.path.abspath(trajectory), trajectory_dest)
   

    
    def convertFromEnsemble(self):
        ensemble = self.inputEnsemble.get()
        ensemble_tmp = self._getTmpPath('ensemble')
        # Una subcarpeta que contenga todos los PDBs
        pdb_folder = os.path.join(ensemble_tmp, 'conformations')
        os.makedirs(pdb_folder, exist_ok=True)

        for i, atom_struct in enumerate(ensemble):
            src = atom_struct.getFileName()
            dst = os.path.join(pdb_folder, f'frame_{i:04d}.pdb')
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        self._ensemblePath = ensemble_tmp


    def callWario(self):
        if self.useSubsequence.get():
            subseq = self._parseSubsequence(self.subsequence.get())
            subseq_arg = ','.join(map(str, subseq.astype(int)))
        else:
            subseq_arg = 'None'  

        plugin_dir = os.path.dirname(os.path.abspath(moma.__file__))
        th_file = os.path.join(plugin_dir, 'protocols', 'contact_thresholds_range.txt')
        runner_dir = os.path.join(plugin_dir, 'protocols', 'scripts')
        extra_path = os.path.abspath(self._getExtraPath())
        ensemble_name = "Protein_sets" 

        mode = self.inputMode.get()
        if mode == INPUT_PATHS:
            structure_folder = os.path.abspath(self._getTmpPath())
        elif mode == INPUT_ENSEMBLE:
            structure_folder = os.path.abspath(self._getTmpPath('ensemble'))
        elif mode == INPUT_MULTIFRAME:
            structure_folder = os.path.abspath(self._getTmpPath())

        #print(">>> DEBUG: structure_folder =", structure_folder)
        #print(">>> DEBUG: th_file =", th_file)
        #print(">>> DEBUG: th_file exists?", os.path.exists(th_file))
        #print(">>> DEBUG: structure_folder contents:", os.listdir(structure_folder))

        cache_path = self._getExtraPath('wario_cache')
        args = (
        ' --ensemble_name "{}"'
        ' --ensemble_path "{}"'
        ' --N_cores {}'
        ' --thresholds "{}"'
        ' --subsequence {}'
        ' --cache_path "{}"'
        ' --extra_path "{}"'
    ).format(ensemble_name, structure_folder, 1, th_file, subseq_arg, cache_path, extra_path)



        #print(">>> DEBUG: args =", args)
        #print(">>> DEBUG: about to call Plugin.runScript...")

        pwchemPlugin.runScript(self, 'wario_runner.py', args, env=MOMA_DIC, cwd=structure_folder, scriptDir=runner_dir)

        #print(">>> DEBUG: Plugin.runScript finished")

    def createOutputStep(self):
        ensemble_folder = os.path.abspath(self._getTmpPath())
        
        centroids_path = self._getExtraPath('centroids')
        if os.path.exists(centroids_path):
            output_set = SetOfAtomStructs.create(self._getPath())
            for pdb_file in sorted(f for f in os.listdir(centroids_path) if f.endswith('.pdb')):
                pdb_path = os.path.join(centroids_path, pdb_file)
                atom_struct = AtomStruct()
                atom_struct.setFileName(pdb_path)
                output_set.append(atom_struct)
            self._defineOutputs(outputCentroids=output_set)

        results_path = os.path.join(ensemble_folder, "results_Protein_sets")
        if os.path.exists(results_path):
            shutil.rmtree(results_path)


    def _parseSubsequence(self, subseq_str):
        ranges = []
        for part in subseq_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                ranges.append(np.arange(int(start), int(end) + 1, 1))
            else:
                ranges.append(np.array([int(part)]))
        return np.concatenate(ranges)