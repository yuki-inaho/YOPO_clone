"""GauCho-3D geometry: SPD(3) Cholesky charts, dual quadrics, and ray solves.

This module is the 3D counterpart of the 2D GauCho ellipse chart used by
``rotated_rtmdet_jax`` (``geometry/ellipse.py``).  The 2D head there regresses
a scale--shape Cholesky factor instead of ``(a, b, theta)`` so that no angle
branch, axis sort, or covariance inverse ever enters training.  The same
principle is extended here to :math:`\\mathbb{S}^3_{++}`.

Canonical 3D state
------------------
An object is ``Ellipsoid3D(t, Sigma)`` with

    E_3(t, Sigma) = { X : (X - t)^T Sigma^{-1} (X - t) <= 1 },

where ``Sigma = L L^T`` is produced from an unconstrained network output via a
Cholesky chart.  Radii, principal axes, Euler angles, and quaternions are
derived diagnostics, never the regression target: eigenvalue coalescence at
spheroids and spheres then removes unobservable pose from the loss instead of
demanding supervision that does not exist.

The only exact 3D -> 2D path is the dual quadric

    Q* ~ [[Sigma - t t^T, -t], [-t^T, -1]],   C* = P Q* P^T,

which is a projective identity for the true silhouette.  Perspective Jacobian
covariance transport, 8-corner projection, and axis-aligned envelopes are all
approximations and are deliberately absent.

All functions are batched over leading dimensions, keep the caller's dtype, and
are written so no eigendecomposition sits on the primary backward path.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "DEFAULT_EPS",
    "scale_shape_cholesky2d",
    "cholesky2d_to_scale_shape",
    "sigma_from_cholesky2d",
    "gaussian_to_ellipse2d",
    "obb_gaussian_to_cholesky2d",
    "cholesky3d_from_raw",
    "cholesky3d_to_raw",
    "scale_shape_cholesky3d",
    "cholesky3d_to_scale_shape",
    "dual_plane_cholesky3d",
    "cholesky3d_to_dual_plane",
    "block_cholesky3d",
    "sigma_from_cholesky",
    "ellipsoid_from_rotation_size",
    "ellipsoid_to_dual_quadric",
    "project_dual_quadric",
    "dual_conic_to_gaussian",
    "project_ellipsoid_dual_quadric",
    "front_margin",
    "ray_frame",
    "ray_ellipsoid_roots",
    "decode_center_from_anchor",
    "ellipsoid_radii_axes",
    "symmetry_class",
]

DEFAULT_EPS = 1e-7

# Raw logarithmic Cholesky entries are clipped before exponentiation so BF16 or
# early-training FP32 forward passes cannot overflow into non-finite shapes.
DEFAULT_LOG_CLIP = 8.0
DEFAULT_SHEAR_CLIP = 8.0

# Floor used only to keep ``log`` finite.  It must sit far below any physical
# Cholesky diagonal: metric fruit radii are centimetre-scale, so a floor tied to
# ``DEFAULT_EPS`` would silently truncate legitimate values.
_LOG_FLOOR = 1e-30

_TRIAXIAL = 0
_SPHEROID = 1
_SPHERE = 2


def _check_last_dim(value: Tensor, size: int, name: str) -> None:
    if value.shape[-1] != size:
        raise ValueError(f"{name} must end in {size} values, got {tuple(value.shape)}")


def _check_matrix(value: Tensor, size: int, name: str) -> None:
    if value.shape[-2:] != (size, size):
        raise ValueError(
            f"{name} must end in shape ({size}, {size}), got {tuple(value.shape)}")


def _as_broadcastable(value: Tensor | float, reference: Tensor, *, name: str) -> Tensor:
    """Return ``value`` as a tensor broadcastable against ``reference``.

    Accepts a python scalar, a 0-d tensor, or a tensor already shaped like
    ``reference``.  Anything else is rejected rather than silently reshaped.
    """
    if not torch.is_tensor(value):
        return reference.new_tensor(float(value))
    tensor = value.to(device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0 or tensor.shape == reference.shape:
        return tensor
    try:
        torch.broadcast_shapes(tensor.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} with shape {tuple(tensor.shape)} is not broadcastable "
            f"against {tuple(reference.shape)}") from error
    return tensor


def _lower_from_entries(
    l11: Tensor, l21: Tensor, l22: Tensor, l31: Tensor, l32: Tensor, l33: Tensor,
) -> Tensor:
    zero = torch.zeros_like(l11)
    row0 = torch.stack((l11, zero, zero), dim=-1)
    row1 = torch.stack((l21, l22, zero), dim=-1)
    row2 = torch.stack((l31, l32, l33), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def cholesky3d_from_raw(
    raw: Tensor,
    *,
    log_clip: float = DEFAULT_LOG_CLIP,
    shear_clip: float = DEFAULT_SHEAR_CLIP,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Build a positive-diagonal lower Cholesky factor from six raw values.

    ``raw = (z1, z2, z3, z4, z5, z6)`` maps to

        L = [[e^z1, 0, 0], [z2, e^z3, 0], [z4, z5, e^z6]].

    Any real input yields an SPD ``Sigma = L L^T`` with no external projection,
    eigenvalue repair, or rejection sampling.
    """
    _check_last_dim(raw, 6, "raw")
    z1 = raw[..., 0].clamp(-log_clip, log_clip)
    z2 = raw[..., 1].clamp(-shear_clip, shear_clip)
    z3 = raw[..., 2].clamp(-log_clip, log_clip)
    z4 = raw[..., 3].clamp(-shear_clip, shear_clip)
    z5 = raw[..., 4].clamp(-shear_clip, shear_clip)
    z6 = raw[..., 5].clamp(-log_clip, log_clip)
    l11 = z1.exp().clamp_min(eps)
    l22 = z3.exp().clamp_min(eps)
    l33 = z6.exp().clamp_min(eps)
    return _lower_from_entries(l11, z2, l22, z4, z5, l33)


def cholesky3d_to_raw(cholesky: Tensor, *, eps: float = DEFAULT_EPS) -> Tensor:
    """Invert :func:`cholesky3d_from_raw` (used by tests and target encoding)."""
    _check_matrix(cholesky, 3, "cholesky")
    z1 = cholesky[..., 0, 0].clamp_min(_LOG_FLOOR).log()
    z3 = cholesky[..., 1, 1].clamp_min(_LOG_FLOOR).log()
    z6 = cholesky[..., 2, 2].clamp_min(_LOG_FLOOR).log()
    return torch.stack(
        (z1, cholesky[..., 1, 0], z3, cholesky[..., 2, 0], cholesky[..., 2, 1], z6),
        dim=-1,
    )


def scale_shape_cholesky3d(
    raw: Tensor,
    rho0: Tensor | float,
    *,
    log_clip: float = DEFAULT_LOG_CLIP,
    shear_clip: float = DEFAULT_SHEAR_CLIP,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Separate metric scale from unit-determinant shape.

    ``raw = (r, u1, u2, v21, v31, v32)`` builds ``rho = rho0 * exp(r)`` and

        B = [[e^u1, 0, 0], [v21, e^u2, 0], [v31, v32, e^{-u1-u2}]],  det B = 1,

    so ``L = rho B`` and ``det Sigma = rho^6``.  Splitting object size from
    anisotropy keeps a size prior (``rho0``) separable from shape learning, the
    same role the FPN cell radius ``rho_Q`` plays in the 2D chart.
    """
    _check_last_dim(raw, 6, "raw")
    r = raw[..., 0].clamp(-log_clip, log_clip)
    u1 = raw[..., 1].clamp(-log_clip, log_clip)
    u2 = raw[..., 2].clamp(-log_clip, log_clip)
    v21 = raw[..., 3].clamp(-shear_clip, shear_clip)
    v31 = raw[..., 4].clamp(-shear_clip, shear_clip)
    v32 = raw[..., 5].clamp(-shear_clip, shear_clip)
    rho0 = _as_broadcastable(rho0, r, name="rho0")
    rho = rho0.clamp_min(_LOG_FLOOR) * r.exp()
    # ``u1 + u2`` is bounded by the clips, so the third diagonal stays finite.
    # The floors below are relative to ``rho`` so they never fire for a clipped
    # chart yet still guarantee a strictly positive diagonal.
    b11 = u1.exp().clamp_min(eps)
    b22 = u2.exp().clamp_min(eps)
    b33 = (-(u1 + u2)).exp().clamp_min(eps)
    l11 = rho * b11
    l22 = rho * b22
    l33 = rho * b33
    return _lower_from_entries(l11, rho * v21, l22, rho * v31, rho * v32, l33)


def cholesky3d_to_scale_shape(
    cholesky: Tensor,
    rho0: Tensor | float,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Invert :func:`scale_shape_cholesky3d`."""
    _check_matrix(cholesky, 3, "cholesky")
    log_l11 = cholesky[..., 0, 0].clamp_min(_LOG_FLOOR).log()
    log_l22 = cholesky[..., 1, 1].clamp_min(_LOG_FLOOR).log()
    log_l33 = cholesky[..., 2, 2].clamp_min(_LOG_FLOOR).log()
    # Metric radii of a fruit are centimetre-scale, so the *product* of the
    # three diagonals underflows far below ``eps`` even though every factor is
    # well above it.  Averaging in log space keeps ``rho`` exact instead of
    # pinning it to a floor.
    log_rho = (log_l11 + log_l22 + log_l33) / 3.0
    rho = log_rho.exp()
    rho0 = _as_broadcastable(rho0, rho, name="rho0")
    r = log_rho - rho0.clamp_min(_LOG_FLOOR).log()
    u1 = log_l11 - log_rho
    u2 = log_l22 - log_rho
    return torch.stack(
        (
            r,
            u1,
            u2,
            cholesky[..., 1, 0] / rho,
            cholesky[..., 2, 0] / rho,
            cholesky[..., 2, 1] / rho,
        ),
        dim=-1,
    )


def dual_plane_cholesky3d(
    raw: Tensor,
    *,
    fix_rc_zero: bool = False,
    log_clip: float = DEFAULT_LOG_CLIP,
    shear_clip: float = DEFAULT_SHEAR_CLIP,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Two orthogonal 2D GauCho factors plus one conditional correlation.

    ``raw = (log_alpha, beta, log_gamma, delta, log_eta, xi)`` describes the two
    2D Cholesky factors ``G12 = [[alpha, 0], [beta, gamma]]`` and
    ``G13 = [[alpha, 0], [delta, eta]]`` that share their first axis, plus
    ``rc = tanh(xi)``.  The resulting

        L_DP = [[alpha, 0, 0], [beta, gamma, 0], [delta, rc*eta, eta*sqrt(1-rc^2)]]

    is a bijection onto SPD(3).  Two orthogonal planes alone supply only
    ``3 + 3 - 1 = 5`` independent degrees of freedom; ``rc`` restores the sixth,
    the residual correlation of the two non-shared axes given the shared one.
    ``fix_rc_zero`` pins ``rc = 0`` regardless of the sixth raw value, giving
    the conditional-independence reduced model of eq (20): two orthogonal 2D
    GauCho factors and nothing else.  It cannot represent a general triaxial
    ellipsoid -- that is the point, it is the ablation that measures how much
    the sixth degree of freedom is worth.
    """
    _check_last_dim(raw, 6, "raw")
    alpha = raw[..., 0].clamp(-log_clip, log_clip).exp()
    beta = raw[..., 1].clamp(-shear_clip, shear_clip)
    gamma = raw[..., 2].clamp(-log_clip, log_clip).exp()
    delta = raw[..., 3].clamp(-shear_clip, shear_clip)
    eta = raw[..., 4].clamp(-log_clip, log_clip).exp()
    rc = (torch.zeros_like(eta) if fix_rc_zero
          else torch.tanh(raw[..., 5]))
    # ``|rc| < 1`` strictly, so the conditional factor stays positive; the floor
    # only guards saturated ``tanh`` in reduced precision.
    l32 = rc * eta
    l33 = eta * (1.0 - rc.square()).clamp_min(eps).sqrt()
    return _lower_from_entries(alpha, beta, gamma, delta, l32, l33)


def cholesky3d_to_dual_plane(
    cholesky: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Invert :func:`dual_plane_cholesky3d`.

    ``eta = sqrt(l32^2 + l33^2)`` and ``rc = l32 / eta``; ``l33 > 0`` forces
    ``|rc| < 1``, so ``xi = atanh(rc)`` is always finite.
    """
    _check_matrix(cholesky, 3, "cholesky")
    alpha = cholesky[..., 0, 0].clamp_min(_LOG_FLOOR)
    gamma = cholesky[..., 1, 1].clamp_min(_LOG_FLOOR)
    l32 = cholesky[..., 2, 1]
    l33 = cholesky[..., 2, 2].clamp_min(_LOG_FLOOR)
    eta = (l32.square() + l33.square()).clamp_min(_LOG_FLOOR).sqrt()
    rc = (l32 / eta).clamp(-1.0 + 1e-12, 1.0 - 1e-12)
    return torch.stack(
        (
            alpha.log(),
            cholesky[..., 1, 0],
            gamma.log(),
            cholesky[..., 2, 0],
            eta.log(),
            torch.atanh(rc),
        ),
        dim=-1,
    )


def block_cholesky3d(
    transverse_cholesky: Tensor,
    coupling: Tensor,
    log_thickness: Tensor,
    *,
    log_clip: float = DEFAULT_LOG_CLIP,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """``2D GauCho + orthogonal lift`` block form.

        L = [[L_perp, 0], [a^T, e^q]]

    ``L_perp`` is a transverse 2x2 GauCho factor (3 dof), ``a`` is the tilt or
    correlation with the depth direction (2 dof), and ``q`` is the conditional
    depth thickness (1 dof).  The total is again exactly 6 dof.
    """
    _check_matrix(transverse_cholesky, 2, "transverse_cholesky")
    _check_last_dim(coupling, 2, "coupling")
    l33 = log_thickness.clamp(-log_clip, log_clip).exp().clamp_min(eps)
    return _lower_from_entries(
        transverse_cholesky[..., 0, 0],
        transverse_cholesky[..., 1, 0],
        transverse_cholesky[..., 1, 1],
        coupling[..., 0],
        coupling[..., 1],
        l33,
    )


def sigma_from_cholesky(cholesky: Tensor) -> Tensor:
    """Return the symmetric ``Sigma = L L^T``."""
    _check_matrix(cholesky, 3, "cholesky")
    sigma = cholesky @ cholesky.transpose(-1, -2)
    return (sigma + sigma.transpose(-1, -2)) * 0.5


def ellipsoid_from_rotation_size(rotation: Tensor, size: Tensor) -> Tensor:
    """Build the ground-truth ``Sigma`` from a pose/extent annotation.

    The dataset stores ``(R, size)``; the canonical state is ``Sigma``.  This is
    a *target* builder only, mirroring ``rboxes_to_inscribed_ellipses`` on the
    2D side: predictions never take this path.
    """
    _check_matrix(rotation, 3, "rotation")
    _check_last_dim(size, 3, "size")
    radii_squared = (size * 0.5).square()
    sigma = rotation @ torch.diag_embed(radii_squared) @ rotation.transpose(-1, -2)
    return (sigma + sigma.transpose(-1, -2)) * 0.5


def ellipsoid_to_dual_quadric(center: Tensor, sigma: Tensor) -> Tensor:
    """Return the 4x4 dual quadric ``Q* ~ [[Sigma - t t^T, -t], [-t^T, -1]]``."""
    _check_last_dim(center, 3, "center")
    _check_matrix(sigma, 3, "sigma")
    top_left = sigma - center.unsqueeze(-1) * center.unsqueeze(-2)
    top_right = -center.unsqueeze(-1)
    top = torch.cat((top_left, top_right), dim=-1)
    bottom = torch.cat(
        (-center.unsqueeze(-2), -torch.ones_like(center[..., :1]).unsqueeze(-2)),
        dim=-1,
    )
    return torch.cat((top, bottom), dim=-2)


def project_dual_quadric(dual_quadric: Tensor, projection: Tensor) -> Tensor:
    """Return ``C* = P Q* P^T`` for a 3x4 camera matrix ``P``."""
    _check_matrix(dual_quadric, 4, "dual_quadric")
    if projection.shape[-2:] != (3, 4):
        raise ValueError(
            f"projection must end in shape (3, 4), got {tuple(projection.shape)}")
    conic = projection @ dual_quadric @ projection.transpose(-1, -2)
    return (conic + conic.transpose(-1, -2)) * 0.5


def dual_conic_to_gaussian(
    dual_conic: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recover ``(mu, S)`` from a dual conic without inverting it.

    Normalizing ``Cbar = -C* / C*_33 = [[A, b], [b^T, -1]]`` gives
    ``mu = -b`` and ``S = A + mu mu^T``.  A camera-front ellipsoid has
    ``C*_33 = Sigma_zz - t_z^2 < 0``, which is checked rather than assumed.

    Note that ``mu`` is the silhouette centre and is *not* the projection of the
    3D centre; for a finite anisotropic ellipsoid the two genuinely differ, so
    comparing against ``pi(t)`` would supervise an identity that does not hold.
    """
    _check_matrix(dual_conic, 3, "dual_conic")
    c33 = dual_conic[..., 2, 2]
    front = c33 < -eps
    safe_c33 = torch.where(front, c33, -torch.ones_like(c33))
    normalized = -dual_conic / safe_c33[..., None, None]
    matrix = normalized[..., :2, :2]
    vector = normalized[..., :2, 2]
    mean = -vector
    sigma = matrix + mean.unsqueeze(-1) * mean.unsqueeze(-2)
    sigma = (sigma + sigma.transpose(-1, -2)) * 0.5
    # 2x2 SPD test without an eigensolver on the backward path.
    trace = sigma[..., 0, 0] + sigma[..., 1, 1]
    determinant = sigma[..., 0, 0] * sigma[..., 1, 1] - sigma[..., 0, 1].square()
    valid = (
        front
        & torch.isfinite(mean).all(dim=-1)
        & torch.isfinite(sigma).all(dim=-1).all(dim=-1)
        & (trace > eps)
        & (determinant > eps * eps)
    )
    return mean, sigma, valid


def project_ellipsoid_dual_quadric(
    center: Tensor,
    sigma: Tensor,
    intrinsic: Tensor,
    *,
    z_min: float = DEFAULT_EPS,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Camera-frame ``(t, Sigma)`` to silhouette ``(mu, S)`` with a validity mask.

    ``P = K [I | 0]`` reduces ``C* = P Q* P^T`` to ``K (Sigma - t t^T) K^T``.

    Validity requires the front margin as well as ``C*_33 < 0``.  The sign test
    alone is not sufficient: ``C*_33 = Sigma_zz - t_z^2`` is negative for an
    ellipsoid *behind* the camera too, as soon as ``|t_z|`` exceeds the object's
    own depth extent, so a fully behind-camera ellipsoid would otherwise project
    to a perfectly well-formed conic and be scored as a detection.
    """
    _check_last_dim(center, 3, "center")
    _check_matrix(sigma, 3, "sigma")
    _check_matrix(intrinsic, 3, "intrinsic")
    dual_conic = intrinsic @ (
        sigma - center.unsqueeze(-1) * center.unsqueeze(-2)
    ) @ intrinsic.transpose(-1, -2)
    dual_conic = (dual_conic + dual_conic.transpose(-1, -2)) * 0.5
    mean, shape, valid = dual_conic_to_gaussian(dual_conic, eps=eps)
    in_front = front_margin(center, sigma, eps=eps) > z_min
    return mean, shape, valid & in_front


def front_margin(center: Tensor, sigma: Tensor, *, eps: float = DEFAULT_EPS) -> Tensor:
    """Return ``m = t_z - sqrt(e_z^T Sigma e_z)``.

    ``m > z_min > 0`` means the whole ellipsoid lies in front of the camera, the
    condition under which the dual-conic normalization is well posed.  Image
    truncation is a separate concern and is not folded into this test.
    """
    _check_last_dim(center, 3, "center")
    _check_matrix(sigma, 3, "sigma")
    return center[..., 2] - sigma[..., 2, 2].clamp_min(eps).sqrt()


def ray_frame(center: Tensor, *, eps: float = DEFAULT_EPS) -> Tensor:
    """Build a stable right-handed frame whose third axis is the centre ray.

    The basis is derived from the camera geometry alone.  Using the image
    ellipse's principal axes instead would leave the frame undefined whenever
    the silhouette is near-circular.
    """
    _check_last_dim(center, 3, "center")
    e3 = center / center.norm(dim=-1, keepdim=True).clamp_min(eps)
    x_axis = torch.zeros_like(e3)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(e3)
    y_axis[..., 1] = 1.0
    use_x = (x_axis * e3).sum(dim=-1, keepdim=True).abs() < 0.9
    seed = torch.where(use_x, x_axis, y_axis)
    e1 = seed - (seed * e3).sum(dim=-1, keepdim=True) * e3
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.cross(e3, e1, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


def ray_ellipsoid_roots(
    directions: Tensor,
    center: Tensor,
    cholesky: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Analytic near/far intersections of ``X(z) = z d`` with the ellipsoid.

    With ``A = Sigma^{-1}`` the intersection solves
    ``a z^2 + b z + c = 0`` for ``a = d^T A d``, ``b = -2 d^T A t``, and
    ``c = t^T A t - 1``.  ``A`` is applied through triangular solves against the
    Cholesky factor, so no explicit inverse is formed.

    Returns ``(z_near, z_far, valid)``; ``valid`` is false where the ray misses
    the ellipsoid or the near root is behind the camera.
    """
    _check_last_dim(directions, 3, "directions")
    _check_last_dim(center, 3, "center")
    _check_matrix(cholesky, 3, "cholesky")

    # y = L^{-1} d and w = L^{-1} t give d^T A d = |y|^2 etc. with A = L^-T L^-1.
    solved_d = torch.linalg.solve_triangular(
        cholesky, directions.unsqueeze(-1), upper=False).squeeze(-1)
    solved_t = torch.linalg.solve_triangular(
        cholesky, center.unsqueeze(-1), upper=False).squeeze(-1)
    a = solved_d.square().sum(dim=-1)
    b = -2.0 * (solved_d * solved_t).sum(dim=-1)
    c = solved_t.square().sum(dim=-1) - 1.0
    discriminant = b.square() - 4.0 * a * c
    hit = (discriminant > 0.0) & (a > eps)
    safe_discriminant = torch.where(
        hit, discriminant, torch.ones_like(discriminant))
    root = safe_discriminant.sqrt()
    safe_a = torch.where(hit, a, torch.ones_like(a))
    z_near = (-b - root) / (2.0 * safe_a)
    z_far = (-b + root) / (2.0 * safe_a)
    valid = hit & (z_near > eps) & torch.isfinite(z_near) & torch.isfinite(z_far)
    return z_near, z_far, valid


def decode_center_from_anchor(
    pixel: Tensor,
    depth_anchor: Tensor,
    residual: Tensor,
    intrinsic: Tensor,
    rho0: Tensor | float,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor]:
    """Decode ``t = t0 + rho0 * B_r * delta_t`` from a robust depth anchor.

    ``t0 = z0 K^{-1} q~`` is the back-projected 2D candidate centre.  The
    residual is expressed in the ray frame so its components stay comparable in
    magnitude regardless of where the object sits in the image.  An invalid
    depth anchor is reported instead of being replaced by a constant depth.
    """
    _check_last_dim(pixel, 2, "pixel")
    _check_matrix(intrinsic, 3, "intrinsic")
    _check_last_dim(residual, 3, "residual")
    homogeneous = torch.cat((pixel, torch.ones_like(pixel[..., :1])), dim=-1)
    rays = torch.linalg.solve(intrinsic, homogeneous.unsqueeze(-1)).squeeze(-1)
    anchor_valid = torch.isfinite(depth_anchor) & (depth_anchor > eps)
    safe_depth = torch.where(
        anchor_valid, depth_anchor, torch.ones_like(depth_anchor))
    t0 = safe_depth.unsqueeze(-1) * rays
    frame = ray_frame(t0, eps=eps)
    rho0 = _as_broadcastable(rho0, depth_anchor, name="rho0")
    center = t0 + rho0.unsqueeze(-1) * (frame @ residual.unsqueeze(-1)).squeeze(-1)
    valid = anchor_valid & torch.isfinite(center).all(dim=-1)
    return center, valid


def ellipsoid_radii_axes(sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Diagnostic decode to ``(radii, axes)``; never on the training path.

    ``eigvalsh``/``eigh`` on a near-spheroid produces arbitrary eigenvectors, so
    this is reserved for reporting and visualization.
    """
    _check_matrix(sigma, 3, "sigma")
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    return eigenvalues.clamp_min(0.0).sqrt(), eigenvectors


def symmetry_class(
    sigma: Tensor,
    *,
    spheroid_ratio: float = 1.1,
    sphere_ratio: float = 1.05,
) -> Tensor:
    """Classify an ellipsoid as triaxial / spheroid / sphere from its radii.

    Pose observability follows the class: a sphere has none, a spheroid has an
    undirected axis only, and only a triaxial ellipsoid has a full (quotient)
    orientation.  Losses and metrics are gated on this rather than supervising
    angles that carry no information.
    """
    if spheroid_ratio < 1.0 or sphere_ratio < 1.0:
        raise ValueError("ratio thresholds must be >= 1")
    if sphere_ratio > spheroid_ratio:
        raise ValueError("sphere_ratio must not exceed spheroid_ratio")
    radii, _ = ellipsoid_radii_axes(sigma)
    smallest = radii[..., 0].clamp_min(DEFAULT_EPS)
    middle = radii[..., 1].clamp_min(DEFAULT_EPS)
    largest = radii[..., 2].clamp_min(DEFAULT_EPS)
    is_sphere = (largest / smallest) < sphere_ratio
    # A spheroid has one distinct radius: either the top two or the bottom two
    # radii coincide while the remaining one separates.
    top_pair_equal = (largest / middle) < spheroid_ratio
    bottom_pair_equal = (middle / smallest) < spheroid_ratio
    is_spheroid = (top_pair_equal ^ bottom_pair_equal) & ~is_sphere
    result = torch.full_like(smallest, float(_TRIAXIAL))
    result = torch.where(is_spheroid, torch.full_like(result, float(_SPHEROID)), result)
    result = torch.where(is_sphere, torch.full_like(result, float(_SPHERE)), result)
    return result.to(torch.long)


TRIAXIAL = _TRIAXIAL
SPHEROID = _SPHEROID
SPHERE = _SPHERE


# ---------------------------------------------------------------------------
# 2D GauCho chart
#
# This is the same scale--shape Cholesky chart the reference 2D implementation
# uses, transplanted from a dense FPN head to a query-based one.  There is no
# stride here, so the role of the cell radius ``rho_Q`` is played by a
# per-query reference radius derived from the predicted box.
# ---------------------------------------------------------------------------


def scale_shape_cholesky2d(
    raw: Tensor,
    reference_radius: Tensor,
    *,
    log_clip: float = DEFAULT_LOG_CLIP,
    shear_clip: float = DEFAULT_SHEAR_CLIP,
) -> Tensor:
    """Build a 2x2 Cholesky factor from ``(r, u, v)`` and a reference radius.

        L = rho_Q e^r [[e^u, 0], [v, e^-u]],   Sigma = L L^T

    ``det L = rho_Q^2 e^{2r}``, so ``r`` alone carries the ellipse area and
    ``(u, v)`` carry its shape.  No angle is produced or consumed anywhere.
    """
    _check_last_dim(raw, 3, "raw")
    r = raw[..., 0].clamp(-log_clip, log_clip)
    u = raw[..., 1].clamp(-log_clip, log_clip)
    v = raw[..., 2].clamp(-shear_clip, shear_clip)
    rho = _as_broadcastable(
        reference_radius, r, name="reference_radius").clamp_min(_LOG_FLOOR)
    scale = rho * r.exp()
    l11 = scale * u.exp()
    l21 = scale * v
    l22 = scale * (-u).exp()
    zero = torch.zeros_like(l11)
    return torch.stack(
        (torch.stack((l11, zero), dim=-1),
         torch.stack((l21, l22), dim=-1)),
        dim=-2,
    )


def cholesky2d_to_scale_shape(
    cholesky: Tensor,
    reference_radius: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Invert :func:`scale_shape_cholesky2d`."""
    _check_matrix(cholesky, 2, "cholesky")
    log_l11 = cholesky[..., 0, 0].clamp_min(_LOG_FLOOR).log()
    log_l22 = cholesky[..., 1, 1].clamp_min(_LOG_FLOOR).log()
    log_scale = (log_l11 + log_l22) / 2.0
    rho = _as_broadcastable(
        reference_radius, log_scale, name="reference_radius")
    return torch.stack(
        (
            log_scale - rho.clamp_min(_LOG_FLOOR).log(),
            log_l11 - log_scale,
            cholesky[..., 1, 0] / log_scale.exp().clamp_min(eps),
        ),
        dim=-1,
    )


def sigma_from_cholesky2d(cholesky: Tensor) -> Tensor:
    """Return the symmetric 2x2 ``Sigma = L L^T``."""
    _check_matrix(cholesky, 2, "cholesky")
    sigma = cholesky @ cholesky.transpose(-1, -2)
    return (sigma + sigma.transpose(-1, -2)) * 0.5


def gaussian_to_ellipse2d(
    mean: Tensor,
    sigma: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Decode to ``(a, b, cx, cy, theta)`` for NMS, evaluation, and drawing.

    This is a *display and metric* path only.  Training never round-trips
    through an angle, which is exactly what removes the boundary discontinuity
    and the axis-swap ambiguity from the objective.
    """
    _check_last_dim(mean, 2, "mean")
    _check_matrix(sigma, 2, "sigma")
    sigma_xx = sigma[..., 0, 0]
    sigma_yy = sigma[..., 1, 1]
    sigma_xy = sigma[..., 0, 1]
    trace = sigma_xx + sigma_yy
    discriminant = (
        (sigma_xx - sigma_yy).square() + 4.0 * sigma_xy.square()
    ).clamp_min(0.0).sqrt()
    major = ((trace + discriminant) * 0.5).clamp_min(eps).sqrt()
    minor = ((trace - discriminant) * 0.5).clamp_min(eps).sqrt()
    theta = 0.5 * torch.atan2(2.0 * sigma_xy, sigma_xx - sigma_yy)
    return torch.stack(
        (major, minor, mean[..., 0], mean[..., 1], theta), dim=-1)


def obb_gaussian_to_cholesky2d(
    compact: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Turn a stored ``(cx, cy, xx, xy, yy)`` target into ``(mu, L, valid)``.

    Degenerate annotations are reported through ``valid`` rather than being
    repaired into a plausible-looking ellipse.
    """
    _check_last_dim(compact, 5, "compact")
    mean = compact[..., :2]
    sigma = torch.stack(
        (compact[..., 2], compact[..., 3], compact[..., 3], compact[..., 4]),
        dim=-1,
    ).reshape(*compact.shape[:-1], 2, 2)
    trace = sigma[..., 0, 0] + sigma[..., 1, 1]
    determinant = (
        sigma[..., 0, 0] * sigma[..., 1, 1] - sigma[..., 0, 1].square())
    valid = (
        torch.isfinite(mean).all(dim=-1)
        & torch.isfinite(sigma).all(dim=-1).all(dim=-1)
        & (trace > eps)
        & (determinant > eps * eps)
    )
    identity = torch.eye(2, dtype=sigma.dtype, device=sigma.device)
    safe_sigma = torch.where(
        valid[..., None, None], sigma, identity.expand_as(sigma))
    cholesky = torch.linalg.cholesky(safe_sigma + eps * identity)
    return mean, cholesky, valid
