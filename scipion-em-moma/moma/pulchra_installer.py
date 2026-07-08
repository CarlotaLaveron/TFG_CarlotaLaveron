import os
from pwem import Plugin as EMPlugin
import pwem

from .constants import PULCHRA_VERSION, PULCHRA_URL, PULCHRA_HOME

#not part of the TFG


class PulchraInstaller(EMPlugin):

    _homeVar = PULCHRA_HOME
    _pathVars = [PULCHRA_HOME]
    _supportedVersions = [PULCHRA_VERSION]
    _url = 'https://github.com/euplotes/pulchra/archive/refs/heads/master.zip'

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(PULCHRA_HOME, f'pulchra-{PULCHRA_VERSION}')

    @classmethod
    def getPulchraBin(cls):
        install_dir = os.path.join(pwem.Config.EM_ROOT, f'pulchra-{PULCHRA_VERSION}')
        return os.path.join(install_dir, 'pulchra')

    @classmethod
    def pulchraExists(cls):
        binary = cls.getPulchraBin()
        return os.path.isfile(binary) and os.access(binary, os.X_OK)

    @classmethod
    def addPulchra(cls, env):
        install_dir = os.path.join(pwem.Config.EM_ROOT, f'pulchra-{PULCHRA_VERSION}')
        binary   = os.path.join(install_dir, 'pulchra')
        zipfile  = os.path.join(install_dir, 'pulchra_master.zip')

        src_dir  = os.path.join(install_dir, 'pulchra-master')
        src_c    = os.path.join(src_dir, 'pulchra.c')
        src_data = os.path.join(src_dir, 'pulchra_data.c')


        commands = [
            (f'mkdir -p {install_dir}',                                   install_dir),
            (f'wget -q -O {zipfile} {PULCHRA_URL}',                       zipfile),
            (f'unzip -q {zipfile} -d {install_dir}',                      src_dir),
            # Compilar desde dentro del directorio fuente para que gcc encuentre los .h
            (f'cd {src_dir} && gcc -O3 -o {binary} pulchra.c pulchra_data.c -lm', binary),
            (f'chmod +x {binary}',                                        binary),
        ]


        env.addPackage(
            'pulchra',
            version=PULCHRA_VERSION,
            tar='void.tgz',
            commands=commands,
            neededProgs=['gcc', 'wget', 'unzip'],
            default=True,
        )