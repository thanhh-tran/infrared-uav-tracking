# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Minimal TrackTrack port for the repository's direct per-frame inference pipeline.

Based on Ultralytics TrackTrack (8.4.63+). This port keeps the official non-ReID
core: NSA Kalman, HMIoU/confidence/corner-angle costs, iterative association,
GMC, and Track-Aware Initialization (TAI).
"""

from collections import deque
from functools import wraps

import numpy as np

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import bbox_ioa

from .basetrack import BaseTrack, TrackState
from .bot_sort import BOTrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import NSAKalmanFilterXYWH

_CORNER_DX_IDX = np.array([0, 0, 2, 2])
_CORNER_DY_IDX = np.array([1, 3, 1, 3])
_LOOSE_NMS_IOU = 0.95
_LOOSE_NMS_DEDUP_IOU = 0.97


def attach_raw_preds_hook(predictor):
    """Capture a second, loose-NMS result from the same raw network output."""
    if hasattr(predictor, "_tracktrack_orig_postprocess"):
        return
    original = predictor.postprocess

    @wraps(original)
    def wrapped(preds, img, orig_imgs, *args, **kwargs):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        loose = original(raw.clone(), img, orig_imgs, iou=_LOOSE_NMS_IOU)
        results = original(preds, img, orig_imgs, *args, **kwargs)
        for tight, alternative in zip(results, loose):
            tight._tracktrack_loose = alternative
        return results

    predictor._tracktrack_orig_postprocess = original
    predictor.postprocess = wrapped


def extract_loose_detections(result):
    """Return `(xywh, confidence, class)` rows dropped by tight NMS."""
    loose = getattr(result, "_tracktrack_loose", None)
    if loose is None:
        return None
    del result._tracktrack_loose
    tight_boxes, loose_boxes = result.boxes, loose.boxes
    if loose_boxes is None or len(loose_boxes) == 0:
        return None
    if tight_boxes is None or len(tight_boxes) == 0:
        data = loose_boxes.data.cpu()
        from ultralytics.utils.ops import xyxy2xywh

        return xyxy2xywh(data[:, :4]).numpy(), data[:, 4].numpy(), data[:, 5].numpy()
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.ops import xyxy2xywh

    maximum_iou = box_iou(loose_boxes.xyxy, tight_boxes.xyxy).max(dim=1).values
    data = loose_boxes.data[maximum_iou < _LOOSE_NMS_DEDUP_IOU].cpu()
    if len(data) == 0:
        return None
    return xyxy2xywh(data[:, :4]).numpy(), data[:, 4].numpy(), data[:, 5].numpy()


def _joint(first, second):
    ids = {track.track_id for track in first}
    return first + [track for track in second if track.track_id not in ids]


def _subtract(first, second):
    ids = {track.track_id for track in second}
    return [track for track in first if track.track_id not in ids]


def _hmiou_distance(tracks, detections):
    if not tracks or not detections:
        shape = (len(tracks), len(detections))
        return np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32)
    boxes_a = np.ascontiguousarray([track.xyxy for track in tracks], dtype=np.float32)
    boxes_b = np.ascontiguousarray([track.xyxy for track in detections], dtype=np.float32)
    iou = bbox_ioa(boxes_a, boxes_b, iou=True)
    overlap = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(
        boxes_a[:, 1:2], boxes_b[:, 1:2].T
    )
    union = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.minimum(
        boxes_a[:, 1:2], boxes_b[:, 1:2].T
    )
    return iou, 1.0 - np.clip(overlap / (union + 1e-9), 0, 1) * iou


def _confidence_distance(tracks, detections):
    if not tracks or not detections:
        return np.ones((len(tracks), len(detections)), dtype=np.float32)
    projected = np.array([2 * track.score - track.prev_score for track in tracks])
    scores = np.array([detection.score for detection in detections])
    return np.abs(projected[:, None] - scores[None, :])


def _angle_distance(tracks, detections, frame_id, pairs, delta_t=3):
    track_indices, detection_indices = pairs
    history = np.stack([tracks[index].get_history_box(frame_id, delta_t) for index in track_indices])
    boxes = np.stack([detections[index].xyxy for index in detection_indices])
    delta = boxes - history
    dx, dy = delta[:, _CORNER_DX_IDX], delta[:, _CORNER_DY_IDX]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-5
    velocities = np.stack([tracks[index].velocity for index in track_indices])
    dot = velocities[:, :, 0] * (dx / norm) + velocities[:, :, 1] * (dy / norm)
    distance = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi
    return distance * np.array([detections[index].score for index in detection_indices])


def _iterative_associate(cost, threshold, reduce_step):
    cost = cost.copy()
    matches = []
    while cost.shape[0] and cost.shape[1]:
        nearest_detection = np.argmin(cost, axis=1)
        nearest_track = np.argmin(cost, axis=0)
        current = [
            [track_index, nearest_detection[track_index]]
            for track_index in range(cost.shape[0])
            if nearest_track[nearest_detection[track_index]] == track_index
            and cost[track_index, nearest_detection[track_index]] < threshold
        ]
        if not current:
            break
        matches.extend(current)
        for track_index, detection_index in current:
            cost[track_index, :] = np.inf
            cost[:, detection_index] = np.inf
        threshold -= reduce_step
    matched_tracks = {track_index for track_index, _ in matches}
    matched_detections = {detection_index for _, detection_index in matches}
    return (
        matches,
        [index for index in range(cost.shape[0]) if index not in matched_tracks],
        [index for index in range(cost.shape[1]) if index not in matched_detections],
    )


def _track_aware_nms(tracks, detections, threshold, new_track_thresh):
    if not detections:
        return []
    scores = np.array([detection.score for detection in detections])
    allowed = scores > new_track_thresh
    n_tracks = len(tracks)
    boxes = np.ascontiguousarray([obj.xyxy for obj in tracks + detections], dtype=np.float32)
    iou = bbox_ioa(boxes, boxes, iou=True)
    if n_tracks:
        allowed &= iou[n_tracks:, :n_tracks].max(axis=1) <= threshold
    detection_iou = iou[n_tracks:, n_tracks:]
    for index in scores.argsort()[::-1]:
        if not allowed[index]:
            continue
        suppress = detection_iou[index] > threshold
        suppress[index] = False
        allowed[suppress] = False
    return allowed.tolist()


class TTSTrack(BOTrack):
    """TrackTrack state with score history and per-corner motion."""

    min_track_len = 3
    _delta_t = 3

    def __init__(self, xywh, score, cls):
        super().__init__(xywh, score, cls)
        self.prev_score = score
        self.velocity = np.zeros((4, 2), dtype=np.float32)
        self._history = deque(maxlen=self._delta_t + 1)

    def get_history_box(self, frame_id, delta_t):
        target = frame_id - delta_t
        for history_frame, box in self._history:
            if history_frame == target:
                return box.copy()
        return self._history[-1][1].copy() if self._history else self.xyxy

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._history.append((frame_id, self.xyxy))
        self.tracklet_len = 0
        self.state = TrackState.Tracked if self.min_track_len <= 1 else TrackState.New
        self.is_activated = frame_id == 1 or self.state == TrackState.Tracked
        self.frame_id = self.start_frame = frame_id

    def _measurement_update(self, new_track, frame_id):
        self.prev_score = self.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score
        )
        self._history.append((frame_id, new_track.xyxy))
        self.score, self.cls, self.idx = new_track.score, new_track.cls, new_track.idx
        self.frame_id = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self._measurement_update(new_track, frame_id)
        self.tracklet_len = 0
        self.state, self.is_activated = TrackState.Tracked, True
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        previous_boxes = [self.get_history_box(frame_id, delta_t) for delta_t in range(1, self._delta_t + 1)]
        self._measurement_update(new_track, frame_id)
        self.tracklet_len += 1
        velocity = np.zeros((4, 2), dtype=np.float32)
        for delta_t, previous in enumerate(previous_boxes, 1):
            delta = new_track.xyxy - previous
            dx, dy = delta[_CORNER_DX_IDX], delta[_CORNER_DY_IDX]
            norm = np.sqrt(dx * dx + dy * dy) + 1e-5
            velocity += np.stack([dx / norm, dy / norm], axis=-1) / delta_t
        self.velocity = velocity / self._delta_t
        if self.state == TrackState.Tracked or self.tracklet_len + 1 >= self.min_track_len:
            self.state, self.is_activated = TrackState.Tracked, True


class TRACKTRACK:
    """TrackTrack TPA + TAI backend compatible with the repository's tracker API."""

    def __init__(self, args, frame_rate=30):
        if getattr(args, "with_reid", False):
            raise NotImplementedError("This minimal TrackTrack port supports with_reid: false only")
        self.args = args
        self.frame_id = 0
        self.tracked_stracks, self.lost_stracks, self.removed_stracks = [], [], []
        self.max_time_lost = args.track_buffer
        self.kalman_filter = NSAKalmanFilterXYWH()
        self.match_thresh = getattr(args, "match_thresh", 0.7)
        self.penalty_p = getattr(args, "penalty_p", 0.2)
        self.penalty_q = getattr(args, "penalty_q", 0.4)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.conf_weight = getattr(args, "conf_weight", 0.1)
        self.angle_weight = getattr(args, "angle_weight", 0.05)
        self.tai_thr = getattr(args, "tai_thr", 0.55)
        self.new_track_thresh = getattr(args, "new_track_thresh", 0.7)
        self.min_track_len = getattr(args, "min_track_len", 3)
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))
        BaseTrack.reset_id()

    def _new_track(self, box, score, cls):
        track = TTSTrack(box, score, cls)
        track.min_track_len = self.min_track_len
        return track

    def _cost_matrix(self, tracks, detections):
        iou, cost = _hmiou_distance(tracks, detections)
        cost += self.conf_weight * _confidence_distance(tracks, detections)
        if iou.size:
            supported = iou > 0.10
            pairs = np.nonzero(supported)
            if len(pairs[0]):
                cost[pairs] += self.angle_weight * _angle_distance(tracks, detections, self.frame_id, pairs)
            cost[~supported] = 1.0
        return np.clip(cost, 0, 1)

    def _apply_gmc(self, image, detections, pools):
        try:
            warp = self.gmc.apply(image, [detection.xyxy for detection in detections])
        except Exception as error:
            LOGGER.warning(f"GMC failed, using identity transform: {error}")
            warp = np.eye(2, 3)
        for pool in pools:
            BOTrack.multi_gmc(pool, warp)

    @staticmethod
    def setup_predictor(predictor):
        attach_raw_preds_hook(predictor)

    def update(self, results, img=None, dets_del=None):
        self.frame_id += 1
        activated, refound, lost, removed = [], [], [], []
        scores = np.asarray(results.conf)
        boxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        boxes = np.concatenate([boxes, np.arange(len(boxes)).reshape(-1, 1)], axis=-1)
        high = scores >= self.args.track_high_thresh
        low = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)
        detections_high = [
            self._new_track(box, score, cls)
            for box, score, cls in zip(boxes[high], scores[high], results.cls[high])
        ]
        detections_low = [
            self._new_track(box, score, cls)
            for box, score, cls in zip(boxes[low], scores[low], results.cls[low])
        ]
        detections_recovered = []
        if dets_del is not None:
            deleted_boxes, deleted_scores, deleted_classes = dets_del
            keep = deleted_scores > self.args.track_high_thresh
            if keep.any():
                indexed_boxes = np.concatenate(
                    [deleted_boxes[keep], -np.ones((int(keep.sum()), 1), dtype=np.float32)], axis=-1
                )
                detections_recovered = [
                    self._new_track(box, score, cls)
                    for box, score, cls in zip(indexed_boxes, deleted_scores[keep], deleted_classes[keep])
                ]

        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            (tracked if track.state == TrackState.Tracked else unconfirmed).append(track)
        pool = _joint(tracked, self.lost_stracks)
        if img is not None and self.gmc.method is not None:
            self._apply_gmc(img, detections_high, [pool, unconfirmed])
        TTSTrack.multi_predict(pool)
        TTSTrack.multi_predict(unconfirmed)

        detections = detections_high + detections_low + detections_recovered
        cost = self._cost_matrix(pool, detections)
        if detections_low:
            start = len(detections_high)
            cost[:, start : start + len(detections_low)] += self.penalty_p
        if detections_recovered:
            cost[:, len(detections_high) + len(detections_low) :] += self.penalty_q
        matches, unmatched_tracks, unmatched_detections = _iterative_associate(
            np.clip(cost, 0, 1), self.match_thresh, self.reduce_step
        )
        for track_index, detection_index in matches:
            track, detection = pool[track_index], detections[detection_index]
            if track.state == TrackState.Tracked:
                track.update(detection, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(detection, self.frame_id)
                refound.append(track)
        for track_index in unmatched_tracks:
            track = pool[track_index]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        leftover = [detections[index] for index in unmatched_detections if index < len(detections_high)]
        if unconfirmed and leftover:
            matches, unmatched_unconfirmed, unmatched_leftover = _iterative_associate(
                self._cost_matrix(unconfirmed, leftover), self.match_thresh, self.reduce_step
            )
            for track_index, detection_index in matches:
                unconfirmed[track_index].update(leftover[detection_index], self.frame_id)
                activated.append(unconfirmed[track_index])
            for track_index in unmatched_unconfirmed:
                unconfirmed[track_index].mark_removed()
                removed.append(unconfirmed[track_index])
            leftover = [leftover[index] for index in unmatched_leftover]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed.append(track)

        active = [track for track in self.tracked_stracks if track.state == TrackState.Tracked] + activated
        for detection, allowed in zip(
            leftover, _track_aware_nms(active, leftover, self.tai_thr, self.new_track_thresh)
        ):
            if allowed:
                detection.activate(self.kalman_filter, self.frame_id)
                activated.append(detection)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = _joint(
            _joint([track for track in self.tracked_stracks if track.state == TrackState.Tracked], activated), refound
        )
        self.lost_stracks = _subtract(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _subtract(self.lost_stracks, removed)
        self.tracked_stracks, self.lost_stracks = self._remove_duplicates(
            self.tracked_stracks, self.lost_stracks
        )
        self.removed_stracks.extend(removed)
        self.removed_stracks = self.removed_stracks[-1000:]
        return np.asarray(
            [track.result for track in self.tracked_stracks if track.is_activated and track.frame_id == self.frame_id],
            dtype=np.float32,
        )

    @staticmethod
    def _remove_duplicates(tracked, lost):
        distance = matching.iou_distance(tracked, lost)
        duplicate_tracked, duplicate_lost = [], []
        for tracked_index, lost_index in zip(*np.where(distance < 0.15)):
            tracked_age = tracked[tracked_index].frame_id - tracked[tracked_index].start_frame
            lost_age = lost[lost_index].frame_id - lost[lost_index].start_frame
            (duplicate_lost if tracked_age > lost_age else duplicate_tracked).append(
                lost_index if tracked_age > lost_age else tracked_index
            )
        return (
            [track for index, track in enumerate(tracked) if index not in duplicate_tracked],
            [track for index, track in enumerate(lost) if index not in duplicate_lost],
        )

    def reset(self):
        self.frame_id = 0
        self.tracked_stracks, self.lost_stracks, self.removed_stracks = [], [], []
        self.kalman_filter = NSAKalmanFilterXYWH()
        BaseTrack.reset_id()
        self.gmc.reset_params()
