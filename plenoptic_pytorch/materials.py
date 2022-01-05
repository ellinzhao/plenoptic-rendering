import torch

from pytorch3d.common.types import Device
from pytorch3d.renderer.utils import TensorProperties

from .settings import N_DIMS


class Materials(TensorProperties):
    """
    A class for storing a batch of material properties. Currently only one
    material per batch element is supported.
    """

    def __init__(
            self,
            ambient_color=(((1,) * N_DIMS),),  # @ellin
            diffuse_color=(((1,) * N_DIMS),),  # @ellin
            specular_color=(((1,) * N_DIMS),),  # @ellin
            shininess=64,
            device: Device = "cpu",
        ) -> None:
        """
        Args:
            ambient_color: RGB ambient reflectivity of the material
            diffuse_color: RGB diffuse reflectivity of the material
            specular_color: RGB specular reflectivity of the material
            shininess: The specular exponent for the material. This defines
                the focus of the specular highlight with a high value
                resulting in a concentrated highlight. Shininess values
                can range from 0-1000.
            device: Device (as str or torch.device) on which the tensors should be located

        ambient_color, diffuse_color and specular_color can be of shape
        (1, 4) or (N, 4). shininess can be of shape (1) or (N).

        The colors and shininess are broadcast against each other so need to
        have either the same batch dimension or batch dimension = 1.
        """
        super().__init__(
            device=device,
            diffuse_color=diffuse_color,
            ambient_color=ambient_color,
            specular_color=specular_color,
            shininess=shininess,
        )
        for n in ["ambient_color", "diffuse_color", "specular_color"]:
            t = getattr(self, n)
            if t.shape[-1] != N_DIMS:  # @ellin
                msg = "Expected %s to have shape (N, %d); got %r"
                raise ValueError(msg % (n, N_DIMS, t.shape))
        if self.shininess.shape != torch.Size([self._N]):
            msg = "shininess should have shape (N); got %r"
            raise ValueError(msg % repr(self.shininess.shape))


    def clone(self):
        other = Materials(device=self.device)
        return super().clone(other)
