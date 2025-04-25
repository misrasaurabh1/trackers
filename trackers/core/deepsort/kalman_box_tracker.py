from typing import Optional, Union

import numpy as np


class DeepSORTKalmanBoxTracker:
    """
    The `DeepSORTKalmanBoxTracker` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position. It also maintains a feature vector for the object, which is
    used to identify the object across frames.

    Attributes:
        tracker_id (int): Unique identifier for the tracker.
        number_of_successful_updates (int): Number of times the object has been
            updated successfully.
        time_since_update (int): Number of frames since the last update.
        state (np.ndarray): State vector of the bounding box.
        F (np.ndarray): State transition matrix.
        H (np.ndarray): Measurement matrix.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        P (np.ndarray): Error covariance matrix.
        features (list[np.ndarray]): List of feature vectors.
        count_id (int): Class variable to assign unique IDs to each tracker.

    Args:
        bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].
        feature (Optional[np.ndarray]): Optional initial feature vector.
    """

    count_id = 0

    @classmethod
    def get_next_tracker_id(cls) -> int:
        """
        Class method that returns the next available tracker ID.

        Returns:
            int: The next available tracker ID.
        """
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        self.tracker_id = -1
        self.number_of_successful_updates = 1
        self.time_since_update = 0
        self.state = np.zeros((8, 1), dtype=np.float32)

        # Directly initialize the relevant state variables to minimize operations
        self.state[:4, 0] = bbox[:4]

        # Set up the Kalman filter matrices
        self._initialize_kalman_filter()

        # Initialize features list more efficiently
        self.features = [feature] if feature is not None else []

    def _initialize_kalman_filter(self) -> None:
        """
        Sets up the matrices for the Kalman filter.
        """
        # State transition matrix (F): 8x8
        # We assume a constant velocity model. Positions are incremented by
        # velocity each step.
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (H): we directly measure x1, y1, x2, y2
        self.H = np.eye(4, 8, dtype=np.float32)  # 4x8

        # Process covariance matrix (Q)
        self.Q = np.eye(8, dtype=np.float32) * 0.01

        # Measurement covariance (R): noise in detection
        self.R = np.eye(4, dtype=np.float32) * 0.1

        # Error covariance matrix (P)
        self.P = np.eye(8, dtype=np.float32)

    def predict(self) -> None:
        """
        Predict the next state of the bounding box (applies the state transition).
        """
        # Predict state
        self.state = self.F @ self.state
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Increase time since update
        self.time_since_update += 1

    def update(self, bbox: np.ndarray) -> None:
        """
        Updates the state with a new detected bounding box.

        Args:
            bbox (np.ndarray): Detected bounding box in the form [x1, y1, x2, y2].
        """
        self.time_since_update = 0
        self.number_of_successful_updates += 1

        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Residual
        measurement = bbox.reshape((4, 1))
        y = measurement - self.H @ self.state

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        identity_matrix = np.eye(8, dtype=np.float32)
        self.P = (identity_matrix - K @ self.H) @ self.P

    def get_state_bbox(self) -> np.ndarray:
        """
        Returns the current bounding box estimate from the state vector.

        Returns:
            np.ndarray: The bounding box [x1, y1, x2, y2].
        """
        # Use slicing for a more efficient data access pattern
        return self.state[:4, 0].astype(float)

    def update_feature(self, feature: np.ndarray):
        self.features.append(feature)

    def get_feature(self) -> Union[np.ndarray, None]:
        """
        Get the mean feature vector for this tracker.

        Returns:
            np.ndarray: Mean feature vector.
        """
        if len(self.features) > 0:
            # Return the mean of all features, thus (in theory) capturing the
            # "average appearance" of the object, which should be more robust
            # to minor appearance changes. Otherwise, the last feature can
            # also be returned like the following:
            # return self.features[-1]
            return np.mean(self.features, axis=0)
        return None
