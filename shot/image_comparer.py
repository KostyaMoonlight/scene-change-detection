import numpy as np
import warnings

class HistogramAnalyzer:
    """
    Analyzes histogram data shaped [parts, channels, bins] and calculates
    various distance-like metrics.
    """

    def __init__(self, data, epsilon=1e-9):
        """
        Initializes the HistogramAnalyzer.

        Args:
            data (np.ndarray): Input data with shape [parts, channels, bins].
            epsilon (float): Small value to add to denominator to avoid division by zero
                             in ratio calculation.

        Raises:
            ValueError: If the input data does not have 3 dimensions.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError(
                f"Input data must be a NumPy array with 3 dimensions "
                f"[parts, channels, bins], but got shape {getattr(data, 'shape', 'N/A')}"
            )

        self.data = data
        self.epsilon = epsilon
        self.parts, self.channels, self.bins = data.shape

        # Optional: Cache for calculated metrics to avoid re-computation
        self._l2_norm = None
        self._l1_norm = None
        self._variation = None
        self._mean_ratio = None

    def _compute_l2_norm(self):
        """Calculates the L2 norm (magnitude) for each [part, channel]."""
        return np.linalg.norm(self.data, axis=2, ord=2)

    def _compute_l1_norm(self):
        """Calculates the L1 norm (sum of abs values) for each [part, channel]."""
        return np.linalg.norm(self.data, axis=2, ord=1)

    def _compute_variation(self):
        """Calculates the sum of absolute differences between adjacent bins."""
        if self.bins < 2:
            # Variation requires at least 2 bins
            return np.zeros((self.parts, self.channels))
        diffs = np.diff(self.data, axis=2)
        return np.sum(np.abs(diffs), axis=2)

    def _compute_mean_ratio(self):
        """Calculates the mean ratio of adjacent bins (d[i+1]/d[i])."""
        if self.bins < 2:
             # Ratio requires at least 2 bins
            return np.zeros((self.parts, self.channels))

        # Select bins for numerator and denominator
        denominator = self.data[:, :, :-1]
        numerator = self.data[:, :, 1:]

        # Calculate ratios, adding epsilon to avoid division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore potential division/invalid warnings
            ratios = numerator / (denominator + self.epsilon)

        # Calculate the mean of the ratios along the (now reduced) bin dimension
        mean_ratios = np.nanmean(ratios, axis=2)
        # Replace any remaining NaNs (e.g., if all bins were zero or only 1 bin)
        return np.nan_to_num(mean_ratios, nan=0.0)

    @property
    def l2_norm(self):
        """Returns the L2 norm metrics [parts, channels], calculates if needed."""
        if self._l2_norm is None:
            self._l2_norm = self._compute_l2_norm()
        return self._l2_norm

    @property
    def l1_norm(self):
        """Returns the L1 norm metrics [parts, channels], calculates if needed."""
        if self._l1_norm is None:
            self._l1_norm = self._compute_l1_norm()
        return self._l1_norm

    @property
    def variation(self):
        """Returns the variation metrics [parts, channels], calculates if needed."""
        if self._variation is None:
            self._variation = self._compute_variation()
        return self._variation

    @property
    def mean_ratio(self):
        """Returns the mean ratio metrics [parts, channels], calculates if needed."""
        if self._mean_ratio is None:
            self._mean_ratio = self._compute_mean_ratio()
        return self._mean_ratio

    def get_all_metrics(self):
        """
        Calculates (if needed) and returns all metrics in a dictionary.

        Returns:
            dict: A dictionary containing all calculated metrics:
                  {'l2_norm': array, 'l1_norm': array,
                   'variation': array, 'mean_ratio': array}
        """
        return {
            'l2_norm': self.l2_norm,
            'l1_norm': self.l1_norm,
            'variation': self.variation,
            'mean_ratio': self.mean_ratio
        }

# --- Example Usage ---
# Create some sample data: 2 parts, 3 channels, 5 bins
# Replace this with your actual data
sample_data = np.array([
    # Part 1
    [
        [10, 20, 30, 20, 10], # Channel 1
        [ 5, 15, 25, 15,  5], # Channel 2
        [ 0, 50, 50,  0,  0]  # Channel 3
    ],
    # Part 2
    [
        [15, 15, 15, 15, 15], # Channel 1
        [ 0,  0, 60,  0,  0], # Channel 2
        [10, 10, 10, 10, 40]  # Channel 3
    ]
])

print(f"Current time: {np.datetime64('now', 's')} UTC") # Adding timestamp as requested indirectly
print("Input data shape:", sample_data.shape)

# Create an instance of the analyzer
try:
    analyzer = HistogramAnalyzer(sample_data)

    # Access metrics using properties
    print("\n--- Accessing metrics via properties ---")
    print("\nL2 Norm (Magnitude) [parts, channels]:")
    print(analyzer.l2_norm)
    print("Shape:", analyzer.l2_norm.shape)

    print("\nL1 Norm (Sum of Absolute Values) [parts, channels]:")
    print(analyzer.l1_norm)
    print("Shape:", analyzer.l1_norm.shape)

    print("\nSum of Absolute Differences (Variation) [parts, channels]:")
    print(analyzer.variation)
    print("Shape:", analyzer.variation.shape)

    print("\nMean Ratio of Adjacent Bins (Relative Change) [parts, channels]:")
    print(analyzer.mean_ratio)
    print("Shape:", analyzer.mean_ratio.shape)

    # Or get all metrics at once
    print("\n--- Accessing all metrics via get_all_metrics ---")
    all_metrics = analyzer.get_all_metrics()
    for name, metric in all_metrics.items():
        print(f"\nMetric: {name}")
        print(metric)

except ValueError as e:
    print(f"Error initializing analyzer: {e}")

# Example with invalid data
# invalid_data = np.random.rand(2, 3) # 2D instead of 3D
# try:
#     invalid_analyzer = HistogramAnalyzer(invalid_data)
# except ValueError as e:
#     print(f"\nError with invalid data: {e}")