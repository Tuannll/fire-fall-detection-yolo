import unittest
from pathlib import Path
import numpy as np
import torch

from src.alerts.telegram_notifier import TelegramNotifier
from src.models.fall_lstm import FallLSTM
from src.tracker.person_tracker import MultiPersonTracker


class TestFallLSTM(unittest.TestCase):
    def test_forward_pass_shape(self):
        batch_size = 2
        seq_len = 30
        input_dim = 102
        model = FallLSTM(input_size=input_dim, hidden=128, n_layers=2, dropout=0.3)
        dummy_input = torch.randn(batch_size, seq_len, input_dim)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (batch_size, 2))

    def test_output_probabilities(self):
        model = FallLSTM(input_size=102, hidden=128, n_layers=2)
        dummy_input = torch.randn(1, 30, 102)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.softmax(logits, dim=1)

        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)


class TestMultiPersonTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = MultiPersonTracker(seq_len=5, fall_confirm=3, cooldown=5)

    def test_multi_track_separation(self):
        # Feed 4 frames for track 1, 5 frames for track 2
        for f in range(4):
            kp = np.ones(51, dtype=np.float32) * f
            seq1 = self.tracker.update_keypoints(1, kp, (10, 10, 50, 50), f)
            self.assertIsNone(seq1)  # not enough frames yet

        for f in range(5):
            kp = np.ones(51, dtype=np.float32) * (f * 2)
            seq2 = self.tracker.update_keypoints(2, kp, (100, 100, 150, 150), f)

        # Track 2 has 5 frames -> must produce feature (5, 102)
        self.assertIsNotNone(seq2)
        self.assertEqual(seq2.shape, (5, 102))

        # Check velocity calculation on track 2
        # kp values: [0, 2, 4, 6, 8] -> deltas: [0, 2, 2, 2, 2]
        expected_vel = np.array([0.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(seq2[:, 51], expected_vel, atol=1e-5)

    def test_fall_confirmation_counter(self):
        self.tracker.get_or_create(1, 0)
        # 1st high prob -> counter = 1
        self.assertFalse(self.tracker.update_fall_evaluation(1, 0.8))
        # 2nd high prob -> counter = 2
        self.assertFalse(self.tracker.update_fall_evaluation(1, 0.9))
        # 3rd high prob -> counter = 3 (reached fall_confirm=3) -> True!
        self.assertTrue(self.tracker.update_fall_evaluation(1, 0.85))

        # While in cooldown -> should be False
        self.assertFalse(self.tracker.update_fall_evaluation(1, 0.9))

    def test_stale_track_cleanup(self):
        self.tracker.update_keypoints(1, np.zeros(51, dtype=np.float32), (0, 0, 10, 10), 10)
        self.assertIn(1, self.tracker.tracks)
        # Advance 200 frames -> stale track must be pruned
        self.tracker.cleanup_stale_tracks(current_frame=220)
        self.assertNotIn(1, self.tracker.tracks)


class TestTelegramNotifier(unittest.TestCase):
    def test_non_blocking_queue(self):
        notifier = TelegramNotifier(bot_token="", chat_id="", max_queue_size=2)
        # Fill queue and test overflow safety
        for i in range(5):
            notifier.send_message(f"Test message {i}")
        notifier.stop()


class TestConfigFile(unittest.TestCase):
    def test_config_structure(self):
        from main import load_config
        cfg = load_config("configs/config.yaml")
        self.assertIn("pipeline", cfg)
        self.assertIn("models", cfg)
        self.assertIn("telegram", cfg)
        self.assertIn("seq_len", cfg["pipeline"])


if __name__ == "__main__":
    unittest.main()
