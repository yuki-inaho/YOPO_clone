"""Closed-form calibration for the YOLO26m-to-YOPO feature boundary."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_finite_matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError(f"{name} must be a non-empty matrix, got {tuple(value.shape)}")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite floating-point values")


def _regularized_gram(gram: Tensor, ridge: float) -> Tensor:
    if not isinstance(ridge, (int, float)) or not 0 <= float(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(ridge))):
        raise ValueError("ridge must be finite and non-negative")
    dimension = gram.shape[0]
    scale = gram.diagonal().mean().abs().clamp_min(torch.finfo(gram.dtype).tiny)
    return gram + float(ridge) * scale * torch.eye(
        dimension, dtype=gram.dtype, device=gram.device
    )


def fit_ridge_projection(
    student_samples: Tensor,
    teacher_samples: Tensor,
    *,
    ridge: float = 1e-4,
) -> Tensor:
    """Fit a bias-free 1x1 projection from paired channel samples.

    Inputs use ``(sample, channel)`` layout.  The returned matrix uses the
    PyTorch convolution layout ``(teacher_channel, student_channel)``.
    ``ridge`` is relative to the mean diagonal of ``X.T @ X`` so its meaning
    remains stable across sample counts and feature scales.
    """

    _validate_finite_matrix("student_samples", student_samples)
    _validate_finite_matrix("teacher_samples", teacher_samples)
    if student_samples.shape[0] != teacher_samples.shape[0]:
        raise ValueError(
            "student and teacher sample counts must match, got "
            f"{student_samples.shape[0]} and {teacher_samples.shape[0]}"
        )
    if student_samples.device != teacher_samples.device:
        raise ValueError("student and teacher samples must use the same device")
    if student_samples.dtype != teacher_samples.dtype:
        raise ValueError("student and teacher samples must use the same dtype")

    gram = student_samples.T @ student_samples
    cross = student_samples.T @ teacher_samples
    return fit_ridge_projection_from_moments(gram, cross, ridge=ridge)


def fit_ridge_projection_from_moments(
    gram: Tensor,
    cross: Tensor,
    *,
    ridge: float = 1e-4,
) -> Tensor:
    """Fit a projection from accumulated ``X.T@X`` and ``X.T@Y`` moments."""

    _validate_finite_matrix("gram", gram)
    _validate_finite_matrix("cross", cross)
    if gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if gram.shape[0] != cross.shape[0]:
        raise ValueError("gram and cross input channels must match")
    if gram.device != cross.device or gram.dtype != cross.dtype:
        raise ValueError("gram and cross must use the same device and dtype")
    try:
        coefficients = torch.linalg.solve(_regularized_gram(gram, ridge), cross)
    except torch.linalg.LinAlgError as error:
        raise ValueError("student feature Gram matrix is not solvable") from error
    projection = coefficients.T.contiguous()
    if not bool(torch.isfinite(projection).all()):
        raise ValueError("fitted projection contains non-finite values")
    return projection


def factor_depth_adapter(
    student_projection: Tensor,
    teacher_projection: Tensor,
    teacher_adapter: Tensor,
    *,
    ridge: float = 1e-6,
) -> Tensor:
    """Factor the teacher's projected depth map through a student projection."""

    _validate_finite_matrix("student_projection", student_projection)
    _validate_finite_matrix("teacher_projection", teacher_projection)
    _validate_finite_matrix("teacher_adapter", teacher_adapter)
    tensors = (student_projection, teacher_projection, teacher_adapter)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all depth-factorization matrices must use the same device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all depth-factorization matrices must use the same dtype")
    if student_projection.shape[0] != teacher_projection.shape[0]:
        raise ValueError("student and teacher projection output channels must match")
    if teacher_projection.shape[1] != teacher_adapter.shape[0]:
        raise ValueError("teacher projection and depth adapter channels do not compose")

    target = teacher_projection @ teacher_adapter
    gram = student_projection @ student_projection.T
    try:
        solved = torch.linalg.solve(_regularized_gram(gram, ridge), target)
    except torch.linalg.LinAlgError as error:
        raise ValueError("student projection has no stable right inverse") from error
    adapter = (student_projection.T @ solved).contiguous()
    if not bool(torch.isfinite(adapter).all()):
        raise ValueError("factored depth adapter contains non-finite values")
    return adapter


__all__ = [
    "factor_depth_adapter",
    "fit_ridge_projection",
    "fit_ridge_projection_from_moments",
]
