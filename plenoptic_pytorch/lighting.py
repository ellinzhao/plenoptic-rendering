import torch
import torch.nn.functional as F

from pytorch3d.common.types import Device
from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast

from .settings import N_DIMS


def diffuse(normals, color, direction) -> torch.Tensor:
    """
    Calculate the diffuse component of light reflection using Lambert's
    cosine law.

    Args:
        normals: (N, ..., 3) xyz normal vectors. Normals and points are
            expected to have the same shape.
        color: (1, 4) or (N, 4) RGB color of the diffuse component of the light.
        direction: (x,y,z) direction of the light

    Returns:
        colors: (N, ..., 4), same shape as the input points.

    The normals and light direction should be in the same coordinate frame
    i.e. if the points have been transformed from world -> view space then
    the normals and direction should also be in view space.

    NOTE: to use with the packed vertices (i.e. no batch dimension) reformat the
    inputs in the following way.

    .. code-block:: python

        Args:
            normals: (P, 3)
            color: (N, 3)[batch_idx, :] -> (P, 3)
            direction: (N, 3)[batch_idx, :] -> (P, 3)

        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes, batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx()
        depending on whether points refers to the vertex coordinates or
        average/interpolated face coordinates.
    """
    # TODO: handle multiple directional lights per batch element.
    # TODO: handle attenuation.

    # Ensure color and location have same batch dimension as normals
    normals, color, direction = convert_to_tensors_and_broadcast(
        normals, color, direction, device=normals.device
    )

    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as normals. Assume first dim = batch dim.
    points_dims = normals.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(points_dims)
    if direction.shape != normals.shape:
        direction = direction.view(expand_dims + (3,))
    if color.shape[:-1] != normals.shape[:-1]:
        color = color.view(expand_dims + (N_DIMS,))  # @ellin

    # Renormalize the normals in case they have been interpolated.
    # We tried to replace the following with F.cosine_similarity, but it wasn't faster.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    angle = F.relu(torch.sum(normals * direction, dim=-1))
    return color * angle[..., None]


def specular(
    points, normals, direction, color, camera_position, shininess
) -> torch.Tensor:
    """
    Calculate the specular component of light reflection.

    Args:
        points: (N, ..., 3) xyz coordinates of the points.
        normals: (N, ..., 3) xyz normal vectors for each point.
        color: (N, 4) RGB color of the specular component of the light.
        direction: (N, 3) vector direction of the light.
        camera_position: (N, 3) The xyz position of the camera.
        shininess: (N)  The specular exponent of the material.

    Returns:
        colors: (N, ..., 4), same shape as the input points.

    The points, normals, camera_position, and direction should be in the same
    coordinate frame i.e. if the points have been transformed from
    world -> view space then the normals, camera_position, and light direction
    should also be in view space.

    To use with a batch of packed points reindex in the following way.
    .. code-block:: python::

        Args:
            points: (P, 3)
            normals: (P, 3)
            color: (N, 3)[batch_idx] -> (P, 3)
            direction: (N, 3)[batch_idx] -> (P, 3)
            camera_position: (N, 3)[batch_idx] -> (P, 3)
            shininess: (N)[batch_idx] -> (P)
        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx().
    """
    # TODO: handle multiple directional lights
    # TODO: attenuate based on inverse squared distance to the light source

    if points.shape != normals.shape:
        msg = "Expected points and normals to have the same shape: got %r, %r"
        raise ValueError(msg % (points.shape, normals.shape))

    # Ensure all inputs have same batch dimension as points
    matched_tensors = convert_to_tensors_and_broadcast(
        points, color, direction, camera_position, shininess, device=points.device
    )
    _, color, direction, camera_position, shininess = matched_tensors

    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as points. Assume first dim = batch dim and last dim = 3.
    points_dims = points.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(points_dims)
    if direction.shape != normals.shape:
        direction = direction.view(expand_dims + (3,))
    if color.shape != normals.shape:
        color = color.view(expand_dims + (N_DIMS,))  # @ellin
    if camera_position.shape != normals.shape:
        camera_position = camera_position.view(expand_dims + (3,))
    if shininess.shape != normals.shape:
        shininess = shininess.view(expand_dims)

    # Renormalize the normals in case they have been interpolated.
    # We tried a version that uses F.cosine_similarity instead of renormalizing,
    # but it was slower.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    cos_angle = torch.sum(normals * direction, dim=-1)
    # No specular highlights if angle is less than 0.
    mask = (cos_angle > 0).to(torch.float32)

    # Calculate the specular reflection.
    view_direction = camera_position - points
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
    reflect_direction = -direction + 2 * (cos_angle[..., None] * normals)

    # Cosine of the angle between the reflected light ray and the viewer
    alpha = F.relu(torch.sum(view_direction * reflect_direction, dim=-1)) * mask
    return color * torch.pow(alpha, shininess)[..., None]


class PointLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2, 0.2),),
        location=((0, 1, 0),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB+IR color of the ambient component
            diffuse_color: RGB+IR color of the diffuse component
            specular_color: RGB+IR color of the specular component
            location: xyz position of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 or 4 element tuple/list or list of lists
            - torch tensor of shape (1, 3 or 4)
            - torch tensor of shape (N, 3 or 4)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
        )
        _validate_light_properties(self)
        if self.location.shape[-1] != 3:
            msg = "Expected location to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def reshape_location(self, points) -> torch.Tensor:
        """
        Reshape the location tensor to have dimensions
        compatible with the points which can either be of
        shape (P, 3) or (N, H, W, K, 3).
        """
        if self.location.ndim == points.ndim:
            # pyre-fixme[7]
            return self.location
        # pyre-fixme[29]
        return self.location[:, None, None, None, :]

    def diffuse(self, normals, points) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )


def _validate_light_properties(obj):
    props = ("ambient_color", "diffuse_color", "specular_color")
    for n in props:
        t = getattr(obj, n)
        if t.shape[-1] != N_DIMS:  # @ellin
            msg = "Expected %s to have shape (N, %d); got %r"
            raise ValueError(msg % (n, N_DIMS, t.shape)) # @ellin
