"""Sequence association and online tracking components."""

from .geometry_context import (
    AssociationResult,
    GeometryContextIdentityHead,
    GeometryExclusion,
    LocalGeometryContext,
    camera_to_world_points,
    matched_identity_loss,
    mutual_nearest_association,
)
from .ellipse_overlay import (
    OpenCVEllipse,
    draw_tracked_ellipse,
    ellipse_to_cv2,
    track_color_bgr,
)
from .online_tracker import (
    AssignmentSolverUnavailable,
    Detection,
    DuplicateAssignmentError,
    OnlineGeometryTracker,
    TrackSnapshot,
    TrackState,
    TrackerConfig,
    TrackerUpdate,
)
from .prediction_cache import (
    PREDICTION_FEATURE_CACHE_SCHEMA,
    TeacherAssignment,
    assign_predictions_to_teachers,
    teacher_obb_envelopes_xyxy,
)
from .sequence_training import (
    PairExample,
    build_pair_examples,
    evaluate_model,
    load_feature_cache,
    sample_pyramid_at_centers,
    train_modes,
)

__all__ = [
    "AssociationResult",
    "AssignmentSolverUnavailable",
    "Detection",
    "DuplicateAssignmentError",
    "GeometryContextIdentityHead",
    "GeometryExclusion",
    "LocalGeometryContext",
    "OnlineGeometryTracker",
    "OpenCVEllipse",
    "PairExample",
    "PREDICTION_FEATURE_CACHE_SCHEMA",
    "TeacherAssignment",
    "TrackSnapshot",
    "TrackState",
    "TrackerConfig",
    "TrackerUpdate",
    "build_pair_examples",
    "assign_predictions_to_teachers",
    "camera_to_world_points",
    "evaluate_model",
    "draw_tracked_ellipse",
    "ellipse_to_cv2",
    "load_feature_cache",
    "matched_identity_loss",
    "mutual_nearest_association",
    "sample_pyramid_at_centers",
    "train_modes",
    "teacher_obb_envelopes_xyxy",
    "track_color_bgr",
]
