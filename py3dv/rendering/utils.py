import torch

from pytorch3d.renderer.cameras import look_at_view_transform


def get_uniform_SO3_RT(num_azimuth, num_elevation, distance, center, device="cpu", add_angle_azi=0, add_angle_ele=0):
    '''
    Get a bunch of camera extrinsics centered towards center with uniform distance in polar coordinates(elevation and azimuth)
    Args:
        num_elevation: int, number of elevation angles, excluding the poles
        num_azimuth: int, number of azimuth angles
        distance: radius of those transforms
        center: center around which the transforms are generated. Needs to be torch.tensor of shape [1, 3]
    Returns:
        rotation: torch.tensor of shape [num_views, 3, 3]
        translation: torch.tensor of shape [num_views, 3]
        Weirdly in pytorch3d y-axis is for world coordinate's up axis
        pytorch3d also has a weird as convention where R is right mulplied, so its actually the inverse of the normal rotation matrix
    '''
    grid = torch.zeros((num_elevation, num_azimuth, 2))  # First channel azimuth, second channel elevation
    azimuth = torch.linspace(0, 360, num_azimuth + 1)[:-1]
    elevation = torch.linspace(-90, 90, num_elevation + 2)[1:-1]
    grid[:, :, 0] = azimuth[None, :]
    grid[:, :, 1] = elevation[:, None]
    grid = grid.view(-1, 2)
    top_down = torch.tensor([[0, -90], [0, 90]])  # [2, 2]
    grid = torch.cat([grid, top_down], dim=0)  # [num_views, 2]
    azimuth = grid[:, 0] + add_angle_azi
    elevation = grid[:, 1] + add_angle_ele

    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=center
    )
    return rotation, translation, azimuth, elevation
