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

V1 = "0.1"
DEFAULT_VERSION = V1
VERSIONS = [V1]

moma_BINARY = "moma_BINARY"
moma_HOME = "moma_HOME"

WARIO_DIC = {'name': 'wario', 'version': '0.1', 'home': 'WARIO_HOME'}
MOMA_DIC = {'name': 'moma', 'version': '0.1', 'home': 'MOMA_HOME'}

MOMA_ENV_ACTIVATION = "MOMA_ENV_ACTIVATION"
MOMA_ENV_NAME = "moma-{version}"

PULCHRA_VERSION = '3.07'
PULCHRA_HOME    = 'PULCHRA_HOME'   
PULCHRA_URL     = 'https://github.com/euplotes/pulchra/archive/refs/heads/master.zip'

CG2ALL_DIC = {'name': 'cg2all','version': '1.2.0'}

PIPPACK_DIC = {'name':'pippack', 'version': '1.0.0', 'home': 'PIPPACK_HOME'}

PIPPACK_ENV_ACTIVATION = 'PIPPACK_ENV_ACTIVATION'
PIPPACK_ENV_NAME = 'pippack-{version}'

CG2ALL_ENV_ACTIVATION = 'CG2ALL_ENV_ACTIVATION'
CG2ALL_ENV_NAME = 'cg2all-{version}'

PLUGIN_NAME    = 'idwbackbone'
PLUGIN_VERSION = '0.1.0'

DIFFPACK_CG2ALL_DIC = {
    'name':    'diffpack-cg2all',
    'version': '1.0.0',
}

DIFFPACK_IDW_DIC = {
    'name':    'diffpack-idw',
    'version': '1.0.0',
}

DIFFPACK_CG2ALL_ENV_ACTIVATION = 'DIFFPACK_CG2ALL_ENV_ACTIVATION'
DIFFPACK_CG2ALL_ENV_NAME = 'diffpack-cg2all-{version}'

DIFFPACK_IDW_ENV_ACTIVATION = 'DIFFPACK_IDW_ENV_ACTIVATION'
DIFFPACK_IDW_ENV_NAME = 'diffpack-idw-{version}'

DEFAULT_K_NEIGHBOURS = 8     
DEFAULT_IDW_POWER    = 2.0   
K_MIN = 1
K_MAX = 50
OUTPUT_STRUCTURES_NAME = 'reconstructedStructures'