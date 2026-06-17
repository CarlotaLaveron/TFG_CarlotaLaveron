import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pyworkflow.viewer import Viewer
from ..protocols.protocol_wario import ProtocolMomaWario


class WarioViewer(Viewer):
    _label = 'Viewer Wario'
    _targets = [ProtocolMomaWario]

    def _visualize(self, obj, **kwargs):
        imgFile = obj._getExtraPath('clusters_2d_Protein_sets.png')
        img = mpimg.imread(imgFile)
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return []