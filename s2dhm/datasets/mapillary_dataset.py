import gin
import numpy as np
import datasets.internal
import collections
from glob import glob
from pathlib import Path
from datasets.base_dataset import BaseDataset


@gin.configurable
class SfDataset(BaseDataset):
    """Sf Dataset Loader.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_intrinsics()

    def _glob_fn(self, root: str):
      return list(root.iterdir())

    def load_intrinsics(self):
        """Load query images intrinsics.


        For RobotCar, all images are rectified and have the same intrinsics.
        Returns:
            filename_to_intrinsics: A dictionary mapping a query filename to
                the intrinsics matrix and distortion coefficients.
        """
        rear_intrinsics = np.reshape(np.array(
            [400.0, 0.0, 508.222931,
             0.0, 400.0, 498.187378,
             0.0, 0.0, 1], dtype=np.float32), (3, 3))
        distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0])
        intrinsics = [(rear_intrinsics, distortion_coefficients)
                      for i in range(len(self._data['query_image_names']))]
        filename_to_intrinsics = dict(
            zip(self._data['query_image_names'], intrinsics))
        self._data['filename_to_intrinsics'] = filename_to_intrinsics


    def key_converter(self, filename: str):
        """Convert an absolute filename the keys format in the 3D files."""
        return str(filename).split('/')[-1]

    def output_converter(self, filename: str):
        """Convert an absolute filename the output prediction format."""
        out='/'.join(str(filename).split('/')[-2:])
        return out

    def _assemble_intrinsics(self, focal, cx, cy, k1, k2):
        """Assemble intrinsics matrix from parameters."""
        intrinsics = np.eye(3)
        intrinsics[0,0] = float(focal)
        intrinsics[1,1] = float(focal)
        intrinsics[0,2] = float(cx)
        intrinsics[1,2] = float(cy)
        distortion_coefficients = np.array([float(k1), float(k2), 0.0, 0.0])
        return intrinsics, distortion_coefficients