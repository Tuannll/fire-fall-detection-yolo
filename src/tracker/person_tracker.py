from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrackState:
    """State tracking for a single person."""
    track_id: int
    buffer: deque = field(default_factory=deque)
    fall_counter: int = 0
    cooldown_timer: int = 0
    fall_prob: float = 0.0
    is_fall_confirmed: bool = False
    last_seen_frame: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None


class MultiPersonTracker:
    """Tracks keypoints and fall state independently for multiple people."""

    def __init__(
        self,
        seq_len: int = 30,
        fall_confirm: int = 12,
        cooldown: int = 30,
        fall_threshold: float = 0.5,
        max_age: int = 150
    ):
        self.seq_len = seq_len
        self.fall_confirm = fall_confirm
        self.cooldown = cooldown
        self.fall_threshold = fall_threshold
        self.max_age = max_age
        self.tracks: Dict[int, TrackState] = {}

    def get_or_create(self, track_id: int, frame_idx: int) -> TrackState:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                buffer=deque(maxlen=self.seq_len),
                last_seen_frame=frame_idx
            )
        track = self.tracks[track_id]
        track.last_seen_frame = frame_idx
        return track

    def update_keypoints(
        self,
        track_id: int,
        kp_51: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """Add keypoints to track buffer. Returns (T, 102) sequence if ready, else None."""
        track = self.get_or_create(track_id, frame_idx)
        track.bbox = bbox
        track.buffer.append(kp_51)

        if len(track.buffer) < self.seq_len:
            return None

        # Calculate 102-dim feature (Keypoints + Temporal Velocity)
        kps_51 = np.stack(list(track.buffer))  # (T, 51)
        vel = np.zeros_like(kps_51)
        vel[1:] = kps_51[1:] - kps_51[:-1]
        seq_102 = np.concatenate([kps_51, vel], axis=1).astype(np.float32)  # (T, 102)
        return seq_102

    def update_fall_evaluation(
        self,
        track_id: int,
        fall_prob: float
    ) -> bool:
        """Update fall counter and check if fall is confirmed."""
        track = self.tracks[track_id]
        track.fall_prob = fall_prob

        if track.cooldown_timer > 0:
            track.cooldown_timer -= 1
            track.is_fall_confirmed = False
            return False

        is_fall_raw = fall_prob > self.fall_threshold
        if is_fall_raw:
            track.fall_counter += 1
        else:
            track.fall_counter = max(0, track.fall_counter - 1)

        if track.fall_counter >= self.fall_confirm:
            track.fall_counter = 0
            track.cooldown_timer = self.cooldown
            track.is_fall_confirmed = True
            return True

        track.is_fall_confirmed = False
        return False

    def cleanup_stale_tracks(self, current_frame: int):
        """Remove tracks that haven't been seen recently to free memory."""
        stale_ids = [
            t_id for t_id, track in self.tracks.items()
            if (current_frame - track.last_seen_frame) > self.max_age
        ]
        for t_id in stale_ids:
            del self.tracks[t_id]
