"""Post-tracker output selection policies for single-target evaluation."""

import numpy as np

from .basetrack import TrackState


class StrongBaselineOutputSelector:
    """Select at most one target after an otherwise unchanged multi-object tracker update."""

    def __init__(self, min_box_area=10.0):
        if min_box_area < 0:
            raise ValueError(f"min_box_area must be non-negative, got {min_box_area}")
        self.min_box_area = float(min_box_area)
        self.prev_id = None

    def reset(self):
        """Reset video-local selection state."""
        self.prev_id = None

    def _eligible(self, track):
        """Apply the official strict area filter in full-frame Kalman coordinates."""
        tlwh = np.asarray(track.tlwh, dtype=np.float64)
        return bool(tlwh.shape == (4,) and np.isfinite(tlwh).all() and tlwh[2] * tlwh[3] > self.min_box_area)

    @staticmethod
    def _result(track):
        """Return one tracker-format row without predicting or mutating the track."""
        return np.asarray(track.result, dtype=np.float32).reshape(1, -1)

    def select(self, tracker):
        """Apply the Strong Baseline active/lost priority after ``tracker.update``."""
        previous_id = self.prev_id
        active_all = [track for track in tracker.tracked_stracks if track.state == TrackState.Tracked]
        lost_all = [track for track in tracker.lost_stracks if track.state == TrackState.Lost]
        active = [track for track in active_all if self._eligible(track)]
        lost = [track for track in lost_all if self._eligible(track)]

        selected = None
        if active:
            selected = max(active, key=lambda track: float(track.score))
            self.prev_id = int(selected.track_id)
        elif lost:
            selected = next((track for track in lost if int(track.track_id) == previous_id), None)
            if selected is None:
                selected = max(lost, key=lambda track: float(track.score))
        return self._result(selected) if selected is not None else np.empty((0, 8), dtype=np.float32)
