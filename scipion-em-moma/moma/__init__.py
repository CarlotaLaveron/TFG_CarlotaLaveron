# **************************************************************************
# *
# * Authors:     you (you@yourinstitution.email)
# *
# * your institution
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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import pwem
from pwem import Plugin as pwemPlugin

from moma.constants import *
from moma.protocols.protocol_wario import ProtocolMomaWario
from moma.protocols.protocol_fixer import ProtocolFixer

__version__ = "0.1"
_logo = "icon.png"
_references = ['']

_plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Plugin(pwemPlugin):
    _url = "https://github.com/scipion-em/scipion-em-moma"

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(MOMA_DIC['home'], cls.getEnvName(MOMA_DIC))
        cls._defineEmVar(PIPPACK_DIC['home'], cls.getEnvName(PIPPACK_DIC))
        cls._defineVar(MOMA_ENV_ACTIVATION, cls.getEnvActivationCommand(MOMA_DIC))
        cls._defineVar(CG2ALL_ENV_ACTIVATION, cls.getEnvActivationCommand(CG2ALL_DIC))
        cls._defineVar(PIPPACK_ENV_ACTIVATION, cls.getEnvActivationCommand(PIPPACK_DIC) )
        cls._defineVar(DIFFPACK_CG2ALL_ENV_ACTIVATION, cls.getEnvActivationCommand(DIFFPACK_CG2ALL_DIC) )
        cls._defineVar(DIFFPACK_IDW_ENV_ACTIVATION, cls.getEnvActivationCommand(DIFFPACK_IDW_DIC) )

    @classmethod
    def defineBinaries(cls, env):
        cls.addWarioPackage(env)
        cls.addCg2allPackage(env) 
        cls.addPulchraPackage(env)
        cls.addPIPPackPackage(env)
        cls.addDiffPackCg2AllPackage(env)
        cls.addDiffPackIDWPackage(env)


    @classmethod
    def addPulchraPackage(cls, env):
        from .pulchra_installer import PulchraInstaller
        PulchraInstaller.addPulchra(env)

    @classmethod
    def getEnvName(cls, packageDictionary):
        """Return the conda environment name for a given package dictionary."""
        return '{}-{}'.format(packageDictionary['name'], packageDictionary['version'])

    @classmethod
    def getEnvActivationCommand(cls, packageDictionary, condaHook=True):
        return '{}conda activate {}'.format(
            cls.getCondaActivationCmd() if condaHook else '',
            cls.getEnvName(packageDictionary)
        )

    @classmethod
    def addCg2allPackage(cls, env):
        env_name = cls.getEnvName(CG2ALL_DIC)
        flag_file = f'{CG2ALL_DIC["name"]}_{CG2ALL_DIC["version"]}_installed'

        install_cmd = (
            f'conda create -y -n {env_name} python=3.10 && '
            f'{cls.getCondaActivationCmd()} conda activate {env_name} && '
            f'pip install requests && '
            f'pip install git+http://github.com/huhlim/cg2all && '
            f'pip install pytz && '
            f'pip uninstall e3nn -y && '
            f'pip install "e3nn==0.4.4" && '
            f'pip uninstall dgl -y && '
            f'pip install dgl==1.0.0 -f https://data.dgl.ai/wheels/repo.html && '
            f'pip uninstall torch nvidia-cudnn-cu11 nvidia-cublas-cu11 '
            f'nvidia-cuda-runtime-cu11 nvidia-cuda-nvrtc-cu11 -y && '
            f'pip install torch==1.13.1+cpu '
            f'--index-url https://download.pytorch.org/whl/cpu && '
            f'touch {flag_file}'
        )

        env.addPackage(
            CG2ALL_DIC['name'],
            version=CG2ALL_DIC['version'],
            tar='void.tgz',
            commands=[(install_cmd, flag_file)],
            neededProgs=['conda', 'pip', 'git'],
            default=True,
        )

    @classmethod
    def addWarioPackage(cls, env):
        """Create the conda environment for moma (wario) and install all dependencies."""

        env_name = cls.getEnvName(MOMA_DIC)

        conda_packages = [
            'numpy==1.23.1',
            'scipy==1.8.1',
            'matplotlib==3.5.2',
            'pandas==1.4.3',
            'Pillow==9.2.0',
            'h5py==3.7.0',
            'networkx==2.8.5',
            'numba==0.57.1',
            'biopython==1.79',
            'mrcfile==1.4.2',
            'MDAnalysis==2.2.0',
            'mdtraj',
            'GridDataFormats==1.0.1',
            'gsd==2.5.3',
            'mmtf-python==1.1.3',
            'faiss-cpu==1.7.2',
            'umap-learn==0.5.3',
            'pynndescent==0.5.10',
            'POT==0.8.2',
            'joblib==1.1.0',
            'scikit-learn',
            'tables',
            'tqdm==4.64.0',
            'seaborn==0.11.2',
            'jupyterlab==3.4.4',
            'ipython==8.4.0',
            'ipywidgets==8.0.5',
            'ipynb==0.5.1',
            'requests==2.28.1',
            'fasteners==0.17.3',
            'msgpack==1.0.4',
            'psutil==5.9.1',
            'pyzmq==23.2.0',
        ]
        
        pip_install = 'pip install ' + ' '.join(f'"{p}"' for p in conda_packages)

        flag_file = '{name}_{version}_installed'.format(**MOMA_DIC)

        install_cmd = (
            f'conda create -y -n {env_name} python=3.8 && '
            # hdbscan==0.8.1 has no Python 3 wheel; install latest 0.8.x from conda-forge
            f'conda install -y -n {env_name} -c conda-forge hdbscan && '
            f'conda install -y -n {env_name} -c conda-forge openmm pdbfixer && '
            f'conda install -y -n {env_name} -c bioconda rosetta && '
            f'{cls.getCondaActivationCmd()} conda activate {env_name} && '
            f'{pip_install} && '
            f'touch {flag_file}'
        )

        env.addPackage(
            MOMA_DIC['name'],
            version=MOMA_DIC['version'],
            tar='void.tgz',
            commands=[(install_cmd, flag_file)],
            neededProgs=['conda'],
            default=True
        )


    @classmethod
    def addPIPPackPackage(cls, env):
        env_name  = cls.getEnvName(PIPPACK_DIC)
        flag_file = f'{PIPPACK_DIC["name"]}_{PIPPACK_DIC["version"]}_installed'
  

        install_cmd = (
            f'conda create -y -n {env_name} python=3.10 && '
            f'{cls.getCondaActivationCmd()} conda activate {env_name} && '
            f'pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu '
            f'--index-url https://download.pytorch.org/whl/cpu && '
            f'pip install torch-geometric torch-scatter torch-sparse '
            f'-f https://data.pyg.org/whl/torch-2.1.0+cpu.html && '
            f'pip install requests && '                                    # ← añadir aquí
            f'pip install omegaconf==2.3.0 hydra-core==1.3.2 lightning==2.0.9 && '
            f'pip install biopython anyio beautifulsoup4 && '
            f'pip uninstall -y setuptools && '
            f'pip install setuptools==67.8.0 && '
            f'git clone https://github.com/Kuhlman-Lab/PIPPack.git PIPPack && '
            f'touch {flag_file}'
        )


        env.addPackage(
            PIPPACK_DIC['name'],
            version=PIPPACK_DIC['version'],
            tar='void.tgz',
            commands=[(install_cmd, flag_file)],
            neededProgs=['conda', 'pip', 'git'],
            default=True,
        )

    @classmethod
    def addDiffPackCg2AllPackage(cls, env):
        env_name  = cls.getEnvName(DIFFPACK_CG2ALL_DIC)
        flag_file = f'{DIFFPACK_CG2ALL_DIC["name"]}_{DIFFPACK_CG2ALL_DIC["version"]}_installed'

        install_cmd = (
            f'conda create -y -n {env_name} python=3.8 && '
            f'{cls.getCondaActivationCmd()} conda activate {env_name} && '
            f'conda install -y pytorch torchvision torchaudio cpuonly -c pytorch && '
            f'conda install -y torchdrug -c milagraph -c conda-forge -c pytorch -c pyg && '
            f'pip install --force-reinstall torch_scatter torch_sparse torch_cluster torch_geometric '
            f'-f https://data.pyg.org/whl/torch-2.4.0+cpu.html && '
            f'pip install biopython==1.77 pyyaml easydict numpy==1.24.4 gdown && '
            f'git clone https://github.com/DeepGraphLearning/DiffPack.git DiffPack && '
            f'mkdir -p DiffPack/model_weights && '
            f'gdown 1tZ9ZOjIxq9SxrkdvbLJyLUBbt2P-mksO '
            f'-O DiffPack/model_weights/gearnet_edge_confidence_converted.pth && '
            # Parchear molecule.py de torchdrug: UpdatePropertyCache(strict=False)
            f'sed -i \'s/mol\\.UpdatePropertyCache()/mol.UpdatePropertyCache(strict=False)/g\' '
            f'$(python -c "import torchdrug; import os; '
            f'print(os.path.join(os.path.dirname(torchdrug.__file__), \'data\', \'molecule.py\'))") && '
            # Parchear molecule.py: SANITIZE_NONE
            f'sed -i \'s/Chem\\.SanitizeMol(mol, sanitizeOps=Chem\\.SanitizeFlags\\.SANITIZE_PROPERTIES)'
            f'/Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)/\' '
            f'$(python -c "import torchdrug; import os; '
            f'print(os.path.join(os.path.dirname(torchdrug.__file__), \'data\', \'molecule.py\'))") && '
            # Parchear yaml: CPU + ruta local de pesos + sanitize=False
            f'python -c "'
            f'import yaml; '
            f'cfg = yaml.safe_load(open(\'DiffPack/config/inference_confidence.yaml\')); '
            f'cfg[\'engine\'][\'gpus\'] = None; '
            f'cfg[\'model_checkpoint\'] = \'model_weights/gearnet_edge_confidence_converted.pth\'; '
            f'cfg[\'test_set\'][\'sanitize\'] = False; '
            f'yaml.dump(cfg, open(\'DiffPack/config/inference_confidence.yaml\', \'w\'))'
            f'" && '
            f'touch {flag_file}'
        )

        env.addPackage(
            DIFFPACK_CG2ALL_DIC['name'],
            version=DIFFPACK_CG2ALL_DIC['version'],
            tar='void.tgz',
            commands=[(install_cmd, flag_file)],
            neededProgs=['conda', 'pip', 'git'],
            default=True,
        )

    @classmethod
    def addDiffPackIDWPackage(cls, env):
        env_name  = cls.getEnvName(DIFFPACK_IDW_DIC)
        flag_file = f'{DIFFPACK_IDW_DIC["name"]}_{DIFFPACK_IDW_DIC["version"]}_installed'

        install_cmd = (
            f'conda create -y -n {env_name} python=3.8 && '
            f'{cls.getCondaActivationCmd()} conda activate {env_name} && '
            f'conda install -y pytorch torchvision torchaudio cpuonly -c pytorch && '
            f'conda install -y torchdrug -c milagraph -c conda-forge -c pytorch -c pyg && '
            f'pip install --force-reinstall torch_scatter torch_sparse torch_cluster torch_geometric '
            f'-f https://data.pyg.org/whl/torch-2.4.0+cpu.html && '
            f'pip install biopython==1.77 pyyaml easydict numpy==1.24.4 gdown && '
            # Parchear molecule.py: UpdatePropertyCache(strict=False)
            f'sed -i \'s/mol\\.UpdatePropertyCache()/mol.UpdatePropertyCache(strict=False)/g\' '
            f'$(python -c "import torchdrug; import os; '
            f'print(os.path.join(os.path.dirname(torchdrug.__file__), \'data\', \'molecule.py\'))") && '
            f'git clone https://github.com/DeepGraphLearning/DiffPack.git DiffPack && '
            f'mkdir -p DiffPack/model_weights && '
            f'gdown 1tZ9ZOjIxq9SxrkdvbLJyLUBbt2P-mksO '
            f'-O DiffPack/model_weights/gearnet_edge_confidence_converted.pth && '
            f'python -c "'
            f'import yaml; '
            f'cfg = yaml.safe_load(open(\'DiffPack/config/inference_confidence.yaml\')); '
            f'cfg[\'engine\'][\'gpus\'] = None; '
            f'cfg[\'model_checkpoint\'] = \'model_weights/gearnet_edge_confidence_converted.pth\'; '
            f'cfg[\'test_set\'][\'sanitize\'] = True; '
            f'yaml.dump(cfg, open(\'DiffPack/config/inference_confidence.yaml\', \'w\'))'
            f'" && '
            f'touch {flag_file}'
        )
        env.addPackage(
            DIFFPACK_IDW_DIC['name'],
            version=DIFFPACK_IDW_DIC['version'],
            tar='void.tgz',
            commands=[(install_cmd, flag_file)],
            neededProgs=['conda', 'pip', 'git'],
            default=True,
        )


_plugin = Plugin
