"""
StreamScope Cross-Section Analysis

Processes StreamScope LIDAR sweep data to analyze stream cross-sections.

Main features:
- Reads measurement logs and parses sweeps, angles, and distances.
- Computes (x, y) coordinates for all points, accounting for water refraction.
- Uses kernel density estimation (KDE) to find centroids for each angle.
- Identifies stream banks based on centroid and water detection logic.
- Calculates cross-sectional area using detected banks and centroids.
- Visualizes raw points, centroids, banks, and overlays manual survey data.
- Outputs summary statistics and saves plots to disk.

Classes:
    LidarPoint: 2D point with distance and uncertainty info.
    LidarMeasurement: All measurements for a single angle.
    LidarSweep: Full sweep of measurements at a timestamp.

Key functions:
    calculate_coordinates(sweeps): Compute coordinates for all points.
    kde_centroids_by_angle(sweep): Find centroids using KDE for each angle.
    process_kde_centroids(sweep): Prune/validate centroids after KDE.
    find_banks(sweep): Detect left/right banks from centroids and water status.
    calculate_cross_sectional_area(sweep): Compute area between banks and centroids.
    graph(sweeps): Visualize all processed data and overlays.
    execute(folder): Main pipeline for reading, processing, and visualizing data.
"""

import os
import re
import matplotlib.pyplot as plt
import datetime
import glob
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.neighbors import KernelDensity
from scipy.ndimage import maximum_filter, label, find_objects
import os
import logging

# Dynamically determine the project root (assume this script is in .../graph/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Logging directory and file
log_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'graph.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LidarPoint:

    """
    Represents a single LIDAR point in 2D space.

    Attributes:
        x (float): X coordinate.
        y (float): Y coordinate.
        distance (float): Distance from origin.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance = math.sqrt(x**2 + y**2)
        self.uncertain = ("None", "None") # ("True", "Raw") or ("False", "Inlier")

    def clear(self):
        """
        Resets the point's coordinates and distance to zero.
        """
        self.x = 0.0
        self.y = 0.0
        self.distance = 0.0

    def as_tuple(self):
        return (self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, LidarPoint):
            return np.allclose([self.x, self.y], [other.x, other.y])
        elif isinstance(other, tuple) or isinstance(other, list):
            return np.allclose([self.x, self.y], other)
        return False

    def __hash__(self):
        return hash((round(self.x, 5), round(self.y, 6)))

class LidarMeasurement:

    """
    Holds all measurements for a single angle sweep.

    Attributes:
        sonar_distance (int): Sonar distance for this measurement.
        angle (float): Angle of the measurement.
        distances (list): List of raw distance readings.
        raw_points (list): List of LidarPoint objects.
        inliers (list): List of inlier points after clustering.
        outliers (list): List of outlier points after clustering.
        certain_centroid (LidarPoint): Centroid considered certain (likely bank or bottom).
        uncertain_centroid (LidarPoint): Centroid considered uncertain.
        in_water (bool): Whether this measurement is in water.
    """

    def __init__(self):
        self.timestamp = None
        self.sonar_distance = 0
        self.angle = 0.0
        self.distances = []
        self.raw_points = []
        self.inliers = []
        self.outliers = []
        self.certain_centroid = None
        self.uncertain_centroid = None
        self.in_water = False

    def clear(self):
        """
        Resets all measurement data to initial state.
        """
        self.timestamp = None
        self.sonar_distance = 0
        self.angle = 0.0
        self.distances.clear()
        self.raw_points.clear()
        self.inliers.clear()
        self.outliers.clear()
        self.certain_centroid = None
        self.uncertain_centroid = None
        self.in_water = False

class LidarSweep:

    """
    Represents a full sweep of laser measurements at a given timestamp.

    Attributes:
        timestamp (datetime): Timestamp of the sweep.
        accelerometer_available (int): Accelerometer status.
        sonar_distances (list): List of sonar distances for the sweep.
        area (float): Calculated area (if applicable).
        stream_width (float): Calculated stream width (if applicable).
        left_bank (LidarPoint): Coordinates of the left bank.
        right_bank (LidarPoint): Coordinates of the right bank.
        measurements (list): List of LidarMeasurement objects.
    """

    def __init__(self):
        self.timestamps = []
        self.accelerometer_available = 0
        self.sonar_distances = []
        self.area = 0.0
        self.area_polygons = {}
        self.stream_width = 0.0
        self.left_bank = None
        self.right_bank = None
        self.measurements = []

    def clear(self):
        """
        Resets all sweep data to initial state.
        """
        self.timestamps.clear()
        self.accelerometer_available = 0
        self.sonar_distances = []
        self.area = 0.0
        self.stream_width = 0.0
        self.left_bank = None
        self.right_bank = None
        self.measurements.clear()

def calculate_cross_sectional_area(sweep):
    """
    Calculates the cross-sectional area using the certain centroids that are between the banks and under the water.
    Calculates area using the minimum, maximum, and average sonar distance as the baseline.

    Args:
        sweep (LidarSweep): The sweep to process.
        plot (bool): If True, plot the cross-section polygon and baselines.

    Returns:
        dict: Dictionary with area estimates for min, max, and avg sonar distances.
    """
    # Ensure banks are defined
    if not (hasattr(sweep, "left_bank") and hasattr(sweep, "right_bank")):
        return {"min_sonar": 0.0, "max_sonar": 0.0, "avg_sonar": 0.0}
    left_bank = sweep.left_bank
    right_bank = sweep.right_bank
    if left_bank is None or right_bank is None:
        return {"min_sonar": 0.0, "max_sonar": 0.0, "avg_sonar": 0.0}

    # Get sonar distances
    sonar_distances = []
    if hasattr(sweep, "sonar_distances") and sweep.sonar_distances:
        sonar_distances = [d * -1 for d in sweep.sonar_distances if d > 0]
    elif hasattr(sweep, "sonar_distance") and sweep.sonar_distance > 0:
        sonar_distances = [sweep.sonar_distance * -1]

    if not sonar_distances:
        return {"min_sonar": 0.0, "max_sonar": 0.0, "avg_sonar": 0.0}

    min_sonar = min(sonar_distances)
    max_sonar = max(sonar_distances)
    avg_sonar = sum(sonar_distances) / len(sonar_distances)

    min_x = min(left_bank.x, right_bank.x)
    max_x = max(left_bank.x, right_bank.x)
    max_sonar_abs = abs(max_sonar)

    # Collect certain centroids sorted by angle (left to right), only those between banks and under water
    centroids = [
        meas.certain_centroid
        for meas in sorted(sweep.measurements, key=lambda m: m.angle)
        if (
            hasattr(meas, "certain_centroid")
            and meas.certain_centroid is not None
            and min_x <= meas.certain_centroid.x <= max_x
            and abs(meas.certain_centroid.y) >= max_sonar_abs
        )
    ]
    if len(centroids) < 2:
        return {"min_sonar": 0.0, "max_sonar": 0.0, "avg_sonar": 0.0}

    # Helper to find intersection of line (p1, p2) with horizontal line y=baseline_y
    def intersection_with_baseline(p1, p2, baseline_y):
        # p1, p2: LidarPoint
        if p1.y == p2.y:
            return None  # Parallel, no intersection
        t = (baseline_y - p1.y) / (p2.y - p1.y)
        if 0 <= t <= 1:
            x = p1.x + t * (p2.x - p1.x)
            return (x, baseline_y)
        return None

    # For each baseline, calculate area between centroids and the baseline (horizontal line at sonar distance)
    def area_with_baseline(baseline_y):
        xs = [pt.x for pt in centroids]
        ys = [pt.y for pt in centroids]

        # Find intersection with baseline for left side
        left_inter = intersection_with_baseline(centroids[0], left_bank, baseline_y)
        # Find intersection with baseline for right side
        right_inter = intersection_with_baseline(centroids[-1], right_bank, baseline_y)

        # Build polygon: left intersection, centroids, right intersection, then close along baseline
        polygon_x = []
        polygon_y = []

        if left_inter is not None:
            polygon_x.append(left_inter[0])
            polygon_y.append(left_inter[1])
        else:
            # Fallback: use left_bank projected to baseline
            polygon_x.append(left_bank.x)
            polygon_y.append(baseline_y)

        polygon_x += xs
        polygon_y += ys

        if right_inter is not None:
            polygon_x.append(right_inter[0])
            polygon_y.append(right_inter[1])
        else:
            polygon_x.append(right_bank.x)
            polygon_y.append(baseline_y)

        # Close polygon along baseline (from right to left)
        polygon_x.append(polygon_x[0])
        polygon_y.append(polygon_y[0])

        area = 0.5 * abs(
            sum(polygon_x[i] * polygon_y[i + 1] - polygon_x[i + 1] * polygon_y[i]
                for i in range(len(polygon_x) - 1))
        )
        return area, polygon_x, polygon_y

    area_min, poly_x_min, poly_y_min = area_with_baseline(min_sonar)
    area_max, poly_x_max, poly_y_max = area_with_baseline(max_sonar)
    area_avg, poly_x_avg, poly_y_avg = area_with_baseline(avg_sonar)

    sweep.area_polygons = {
        "min": (poly_x_min, poly_y_min),
        "max": (poly_x_max, poly_y_max),
        "avg": (poly_x_avg, poly_y_avg)
    }

    sweep.area = area_avg / 1000000  # Convert mm^2 to m^2

def kde_centroids_by_angle(sweep, bandwidth=50):

    for idx, meas in enumerate(sweep.measurements):
        coords = np.array([pt.as_tuple() for pt in meas.raw_points if not np.isnan(pt.x) and not np.isnan(pt.y)])
        if len(coords) == 0:
            meas.certain_centroid = None
            meas.uncertain_centroid = None
            continue

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(coords)

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        log_density = kde.score_samples(grid_points)
        log_density_2d = log_density.reshape(xx.shape)

        # Find peaks in the density map
        neighborhood_size = 10
        max_filtered = maximum_filter(log_density_2d, size=neighborhood_size)
        peaks_mask = (log_density_2d == max_filtered)
        labeled, num_features = label(peaks_mask)
        peak_indices = []
        for i in range(1, num_features + 1):
            slice_x, slice_y = find_objects(labeled == i)[0]
            peak_region = log_density_2d[slice_x, slice_y]
            max_idx = np.unravel_index(np.argmax(peak_region), peak_region.shape)
            peak_indices.append((slice_x.start + max_idx[0], slice_y.start + max_idx[1]))

        if not peak_indices:
            meas.certain_centroid = None
            meas.uncertain_centroid = None
            continue

        # Sort peaks by density (highest first)
        peak_indices = sorted(peak_indices, key=lambda idx: log_density_2d[idx], reverse=True)

        # First (highest) peak as certain centroid
        i, j = peak_indices[0]
        x_certain = xx[i, j]
        y_certain = yy[i, j]
        meas.certain_centroid = LidarPoint(x_certain, y_certain)

        # Second highest peak as uncertain centroid, if it exists and not near sonar distance
        sonar_dist = sweep.sonar_distances[0] if hasattr(sweep, "sonar_distances") and sweep.sonar_distances else 0
        uncertain_centroid = None
        for idx2 in peak_indices[1:]:
            i2, j2 = idx2
            x_uncertain = xx[i2, j2]
            y_uncertain = yy[i2, j2]
            if abs(abs(y_uncertain) - abs(sonar_dist)) > 100:
                uncertain_centroid = LidarPoint(x_uncertain, y_uncertain)
                break
        meas.uncertain_centroid = uncertain_centroid

def process_kde_centroids(sweep):
    """
    Prunes certain centroids if they are shorter than the sonar distance and between the left and right banks,
    or if their y value is within +/- 3 of the sonar distance AND the angle is within +/- 3 degrees.

    Args:
        sweep (LidarSweep): The sweep to process.
    """
    if not (hasattr(sweep, "left_bank") and hasattr(sweep, "right_bank")):
        return

    left_bank = sweep.left_bank
    right_bank = sweep.right_bank

    if left_bank is None or right_bank is None:
        return

    # Use the first sonar distance as reference (if available)
    sonar_dist = 0
    if hasattr(sweep, "sonar_distances") and sweep.sonar_distances:
        sonar_dist = sweep.sonar_distances[0]

    min_x = min(left_bank.x, right_bank.x)
    max_x = max(left_bank.x, right_bank.x)

    for meas in sweep.measurements:
        cc = getattr(meas, "certain_centroid", None)
        if cc is not None:
            # Prune if centroid is between banks and y is less than sonar distance
            if min_x < cc.x < max_x and abs(cc.y) < sonar_dist:
                meas.certain_centroid = None
                continue
            
            # Remove certain centroids if they are within 50mm of the sonar distance
            # and there exists any raw point at least 50mm further away (in y)

            # Compute adaptive threshold based on average distance of certain centroids to sonar distance
            all_cc_y = [
                m.certain_centroid.y
                for m in sweep.measurements
                if hasattr(m, "certain_centroid") and m.certain_centroid is not None
            ]
            if all_cc_y:
                avg_dist_to_sonar = np.mean([abs(abs(y) - sonar_dist) for y in all_cc_y])
                threshold = max(50, min(400, avg_dist_to_sonar))  # Clamp between 50 and 400 mm
            else:
                threshold = 200  # Fallback default

            if abs(abs(cc.y) - sonar_dist) <= threshold and abs(meas.angle) <= 15:
                meas.certain_centroid = None
            


def calculate_coordinates(sweeps):

    """
    Computes (x, y) coordinates for all laser measurements in the provided sweeps,
    accounting for refraction at the water surface.

    Args:
        sweeps (list): List of LidarSweep objects.
    """

    n1 = 1.00
    n2 = 1.33

    for sweep in sweeps:
        pass
        for meas in sweep.measurements:
            water_level = meas.sonar_distance
            rad_angle = math.radians(meas.angle)
            for i, dist in enumerate(meas.distances):
                if 500 < dist < 5000:

                    da = water_level / math.cos(rad_angle)
                    db = dist - da

                    if db <= 0:
                        xa = math.sin(rad_angle) * dist
                        ya = math.cos(rad_angle) * dist
                        point = LidarPoint(-1 * xa, -1 * ya)
                        meas.raw_points.append(point)
                    else:
                        beta = math.asin(n1 * math.sin(rad_angle) / n2)
                        xa = math.sin(rad_angle) * da
                        xb = math.sin(beta) * db / n2
                        yb = math.cos(beta) * db / n2
                        point = LidarPoint(-1 * (xa + xb), -1 * (water_level + yb))
                        meas.raw_points.append(point)
                        meas.in_water = True

def find_banks(sweep):

    """
    Identifies the left and right stream banks based on centroids and water detection.

    Args:
        sweep (LidarSweep): The sweep to process.
    """

    sorted_measurements = sorted(sweep.measurements, key=lambda m: m.angle)
    n = len(sorted_measurements)

    def is_underwater(meas):
        return meas.in_water

    left_bank = None
    left_angle = None
    for idx, meas in enumerate(sorted_measurements):
        if is_underwater(meas):
            prev_idx = idx - 1
            if prev_idx >= 0:
                prev_meas = sorted_measurements[prev_idx]
                if hasattr(prev_meas, "certain_centroid") and prev_meas.certain_centroid is not None:
                    left_bank = prev_meas.certain_centroid
                    left_angle = prev_meas.angle
                elif hasattr(prev_meas, "uncertain_centroid") and prev_meas.uncertain_centroid is not None:
                    left_bank = prev_meas.uncertain_centroid
                    left_angle = prev_meas.angle
                else:
                    left_bank = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None else meas.uncertain_centroid
                    left_angle = meas.angle
            else:
                left_bank = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None else meas.uncertain_centroid
                left_angle = meas.angle
            break

    right_bank = None
    right_angle = None
    n = len(sorted_measurements)
    found_underwater = False
    underwater_angle = None
    underwater_centroid = None

    for idx in reversed(range(n)):
        meas = sorted_measurements[idx]
        if not found_underwater:
            if is_underwater(meas):
                found_underwater = True
                underwater_angle = meas.angle
                underwater_centroid = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None else meas.uncertain_centroid
    
    for idx in reversed(range(n)):
        meas = sorted_measurements[idx]
        if underwater_angle is not None and meas.angle > underwater_angle:
            candidate_centroid = None
            if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None:
                candidate_centroid = meas.certain_centroid
            elif hasattr(meas, "uncertain_centroid") and meas.uncertain_centroid is not None:
                candidate_centroid = meas.uncertain_centroid
            closest_candidate = None
            closest_angle = None
            min_dist = float('inf')
            for idx2 in reversed(range(n)):
                meas2 = sorted_measurements[idx2]
                if meas2.angle > underwater_angle:
                    cand_centroid = None
                    if hasattr(meas2, "certain_centroid") and meas2.certain_centroid is not None:
                        cand_centroid = meas2.certain_centroid
                    elif hasattr(meas2, "uncertain_centroid") and meas2.uncertain_centroid is not None:
                        cand_centroid = meas2.uncertain_centroid
                    if cand_centroid and underwater_centroid and abs(cand_centroid.x) > abs(underwater_centroid.x):
                        dist = abs(cand_centroid.x - underwater_centroid.x)
                        if dist < min_dist:
                            min_dist = dist
                            closest_candidate = cand_centroid
                            closest_angle = meas2.angle
            if closest_candidate:
                right_bank = closest_candidate
                right_angle = closest_angle

    sweep.left_bank = left_bank
    sweep.right_bank = right_bank

    for meas in sweep.measurements:
        if meas.uncertain_centroid is not None:
            sonar_dist = sweep.sonar_distances[0] if hasattr(sweep, "sonar_distances") and sweep.sonar_distances else 0
            uc = meas.uncertain_centroid
            if sweep.left_bank and sweep.right_bank:
                min_x = min(sweep.left_bank.x, sweep.right_bank.x)
                max_x = max(sweep.left_bank.x, sweep.right_bank.x)
                if uc.x < min_x or uc.x > max_x:
                    meas.uncertain_centroid = None
                    continue
            if abs(uc.y) < sonar_dist:
                meas.uncertain_centroid = None
        
        if meas.certain_centroid is not None:
            sonar_dist = sweep.sonar_distances[0] if hasattr(sweep, "sonar_distances") and sweep.sonar_distances else 0
            cc = meas.certain_centroid
            if sweep.left_bank and sweep.right_bank:
                min_x = min(sweep.left_bank.x, sweep.right_bank.x)
                max_x = max(sweep.left_bank.x, sweep.right_bank.x)
                if cc.x > min_x and cc.x < max_x and abs(cc.y) < sonar_dist:
                    meas.certain_centroid = None
                    continue

    # Rechoosing uncertain centroids if they are too close to certain centroids
    for meas in sweep.measurements:
        if meas.uncertain_centroid is not None and meas.certain_centroid is not None:
            uc = meas.uncertain_centroid
            cc = meas.certain_centroid
            if abs(uc.y - cc.y) < 100:
                if meas.inliers:
                    farthest_inlier = max(
                        (pt for pt in meas.inliers if abs(pt.y) > sweep.sonar_distances[0] and pt.x > min(sweep.left_bank.x, sweep.right_bank.x) and pt.x < max(sweep.left_bank.x, sweep.right_bank.x)),
                        key=lambda pt: abs(pt.y),
                        default=None
                    )
                    if abs(farthest_inlier.y - cc.y) > 100:
                        meas.uncertain_centroid = farthest_inlier
                    else:
                        farthest_raw = max(
                            (pt for pt in meas.raw_points if abs(pt.y) > sweep.sonar_distances[0] and pt.x > min(sweep.left_bank.x, sweep.right_bank.x) and pt.x < max(sweep.left_bank.x, sweep.right_bank.x)),
                            key=lambda pt: abs(pt.y),
                            default=None
                        )
                        meas.uncertain_centroid = farthest_raw if farthest_raw else None

    # Print the number of angles that were in_water
    num_in_water = sum(1 for meas in sweep.measurements if meas.in_water)
    print(f"Number of angles that were in_water: {num_in_water}")

def graph(sweeps):
    """
    Visualizes the processed data, including raw points, centroids, banks, and overlays USGS cross-section data.
    Plots each data type separately and all together, saving each plot to the appropriate folder.
    Graph filenames are prefixed with the date for clarity.
    """

    # Helper to get date folder and time range from sweeps
    def get_date_and_times(sweeps):
        for sweep in sweeps:
            timestamps = getattr(sweep, "timestamps", [])
            if timestamps:
                date_str = timestamps[0].strftime("%m/%d/%Y")
                start_time = timestamps[0].strftime("%H:%M")
                end_time = timestamps[-1].strftime("%H:%M")
                return date_str, start_time, end_time
        return "unknown_date", "", ""
    
    prefix = datetime.datetime.now().strftime("%m%d%Y")
    date_dir = os.path.join(PROJECT_ROOT, "data", prefix, "graphs")
    os.makedirs(date_dir, exist_ok=True)

    date_str, start_time, end_time = get_date_and_times(sweeps)
    title_prefix = f"StreamScope Cross-Section ({date_str}, {start_time}-{end_time} UTC)"

    # Prepare data
    raw_x, raw_y = [], []
    centroid_x, centroid_y = [], []
    uncertain_centroid_x, uncertain_centroid_y = [], []
    angle_centroid_pairs = []
    angle_uncertain_pairs = []

    for sweep in sweeps:
        for measurement in sweep.measurements:
            for point in getattr(measurement, "raw_points", []):
                raw_x.append(-point.x)
                raw_y.append(-point.y)
            if hasattr(measurement, "certain_centroid") and measurement.certain_centroid is not None:
                centroid_x.append(-measurement.certain_centroid.x)
                centroid_y.append(-measurement.certain_centroid.y)
                angle_centroid_pairs.append((measurement.angle, measurement.certain_centroid))
            if hasattr(measurement, "uncertain_centroid") and measurement.uncertain_centroid is not None:
                uncertain_centroid_x.append(-measurement.uncertain_centroid.x)
                uncertain_centroid_y.append(-measurement.uncertain_centroid.y)
                angle_uncertain_pairs.append((measurement.angle, measurement.uncertain_centroid))

    # Banks
    sweep = sweeps[0]
    left_bank = getattr(sweep, "left_bank", None)
    right_bank = getattr(sweep, "right_bank", None)
    left_bank_plot_x = -left_bank.x if left_bank is not None else None
    right_bank_plot_x = -right_bank.x if right_bank is not None else None

    # Sonar distances
    sonar_distances = [d for sweep in sweeps if hasattr(sweep, "sonar_distances") for d in sweep.sonar_distances]
    min_sonar = min(sonar_distances) if sonar_distances else None
    max_sonar = max(sonar_distances) if sonar_distances else None

    # Manual survey overlay
    csv_path = os.path.join(os.path.dirname(__file__), "Falls_Creek_Manual_Survey.csv")
    manual_xs, manual_ys = None, None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().rstrip(',') for c in df.columns]
        df = df[pd.to_numeric(df["Horizontal Distance"], errors="coerce").notnull()]
        df = df[pd.to_numeric(df["Height"], errors="coerce").notnull()]
        xs = df["Horizontal Distance"].astype(float).values
        ys = df["Height"].astype(float).values
        xs_mm = xs * 304.8
        ys_mm = ys * 304.8
        first_in_water_idx = None
        last_in_water_idx = None
        if "Description" in df.columns:
            for i, desc in enumerate(df["Description"].astype(str)):
                if "First In Water" in desc and first_in_water_idx is None:
                    first_in_water_idx = i
                if "Last In Water" in desc:
                    last_in_water_idx = i
        if first_in_water_idx is None:
            first_in_water_idx = 0
        if last_in_water_idx is None:
            last_in_water_idx = len(xs_mm) - 1
        mid_idx = (first_in_water_idx + last_in_water_idx) // 2
        x0 = xs_mm[mid_idx]
        y0 = ys_mm[mid_idx]
        manual_xs = xs_mm - x0 - 1010
        manual_ys = ys_mm - y0 + 3025
        sorted_indices = np.argsort(manual_xs)
        manual_xs = manual_xs[sorted_indices]
        manual_ys = manual_ys[sorted_indices]

    # Helper for consistent plot
    def plot_base(ax, subtitle):
        if left_bank_plot_x is not None:
            ax.axvline(x=left_bank_plot_x, color='brown', linestyle='-.', linewidth=2, label="Left Bank")
        if right_bank_plot_x is not None:
            ax.axvline(x=right_bank_plot_x, color='purple', linestyle='-.', linewidth=2, label="Right Bank")
        if min_sonar is not None and max_sonar is not None:
            ax.axhspan(max_sonar, min_sonar, color='red', alpha=0.15, label="Sonar Distance Range")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(10000, 0)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.set_title(f"{title_prefix}\n{subtitle}")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower left')



    # Plot raw points
    fig, ax = plt.subplots(figsize=(8, 8))
    if raw_x and raw_y:
        ax.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")
    plot_base(ax, "Raw Points Only")
    plt.tight_layout()
    plt.savefig(os.path.join(date_dir, f"{prefix}raw_points.png"))
    plt.close(fig)

    # Plot certain centroids (always include raw points)
    fig, ax = plt.subplots(figsize=(8, 8))
    if raw_x and raw_y:
        ax.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")
    if centroid_x and centroid_y:
        ax.scatter(centroid_x, centroid_y, s=80, color='green', marker='X', label="Certain Centroids", zorder=90, edgecolors='black', linewidths=1.2)
    if angle_centroid_pairs:
        angle_centroid_pairs.sort(key=lambda x: x[0])
        cx = [(c.x) for _, c in angle_centroid_pairs]
        cy = [-(c.y) for _, c in angle_centroid_pairs]
        ax.plot(cx, cy, color='lime', linewidth=2, zorder=1, label="Certain Centroids Contour")
    plot_base(ax, "Certain Centroids + Raw Points")
    plt.tight_layout()
    plt.savefig(os.path.join(date_dir, f"{prefix}certain_centroids.png"))
    plt.close(fig)

    # Plot uncertain centroids (always include raw points, add contour)
    fig, ax = plt.subplots(figsize=(8, 8))
    if raw_x and raw_y:
        ax.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")
    if uncertain_centroid_x and uncertain_centroid_y:
        ax.scatter(uncertain_centroid_x, uncertain_centroid_y, color='orange', marker='X', s=50, edgecolors='black', label="Uncertain Centroids")
    if angle_uncertain_pairs:
        angle_uncertain_pairs.sort(key=lambda x: x[0])
        ucx = [(c.x) for _, c in angle_uncertain_pairs]
        ucy = [-(c.y) for _, c in angle_uncertain_pairs]
        ax.plot(ucx, ucy, color='orange', linewidth=2, linestyle='--', zorder=1, label="Uncertain Centroids Contour")
    plot_base(ax, "Uncertain Centroids + Raw Points")
    plt.tight_layout()
    plt.savefig(os.path.join(date_dir, f"{prefix}uncertain_centroids.png"))
    plt.close(fig)

    # Plot manual survey if available (always include raw points)
    if manual_xs is not None and manual_ys is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        if raw_x and raw_y:
            ax.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")
        ax.plot(-manual_xs, manual_ys, color='blue', linewidth=2, label="Manual Survey Contour", zorder=51)
        ax.scatter(-manual_xs, manual_ys, color='blue', s=80, marker='X', edgecolors='black', linewidths=1.2, label="Manual Survey Points", zorder=50)
        plot_base(ax, "Manual Survey + Raw Points")
        plt.tight_layout()
        plt.savefig(os.path.join(date_dir, f"{prefix}manual_survey.png"))
        plt.close(fig)

    # Plot all together
    fig, ax = plt.subplots(figsize=(8, 8))
    if raw_x and raw_y:
        ax.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")
    if centroid_x and centroid_y:
        ax.scatter(centroid_x, centroid_y, s=80, color='green', marker='X', label="Certain Centroids", zorder=90, edgecolors='black', linewidths=1.2)
    if angle_centroid_pairs:
        angle_centroid_pairs.sort(key=lambda x: x[0])
        cx = [(c.x) for _, c in angle_centroid_pairs]
        cy = [-(c.y) for _, c in angle_centroid_pairs]
        ax.plot(cx, cy, color='lime', linewidth=2, zorder=1, label="Certain Centroids Contour")
    if uncertain_centroid_x and uncertain_centroid_y:
        ax.scatter(uncertain_centroid_x, uncertain_centroid_y, color='orange', marker='X', s=50, edgecolors='black', label="Uncertain Centroids")
    if angle_uncertain_pairs:
        angle_uncertain_pairs.sort(key=lambda x: x[0])
        ucx = [(c.x) for _, c in angle_uncertain_pairs]
        ucy = [-(c.y) for _, c in angle_uncertain_pairs]
        ax.plot(ucx, ucy, color='orange', linewidth=2, linestyle='--', zorder=1, label="Uncertain Centroids Contour")
    if manual_xs is not None and manual_ys is not None:
        ax.plot(-manual_xs, manual_ys, color='blue', linewidth=2, label="Manual Survey Contour", zorder=51)
        ax.scatter(-manual_xs, manual_ys, color='blue', s=80, marker='X', edgecolors='black', linewidths=1.2, label="Manual Survey Points", zorder=50)
    plot_base(ax, "All Data Combined")
    plt.tight_layout()
    plt.savefig(os.path.join(date_dir, f"{prefix}all_combined.png"))
    plt.close(fig)

def execute():
    """
    Main data processing and visualization pipeline.
    Reads log files, processes sweeps, computes coordinates, removes outliers,
    finds centroids and banks, and visualizes the results.
    """
    today = datetime.datetime.now().strftime("%m%d%Y")
    file_pattern = os.path.join(PROJECT_ROOT, "data", today, "*.txt")
    file_list = glob.glob(file_pattern)
    print(f"Found {len(file_list)} files matching pattern: {file_pattern}")
    all_sweeps = [] 

    num_distances = 0

    for file_name in file_list:
        try:
            with open(file_name, 'r') as file:
                print(f"Processing file: {file_name}")
                lines = file.readlines()
                sweep = LidarSweep()
                
                for i, line in enumerate(lines):
                    line = line.strip()

                    if line.startswith("Date:") and "Time:" in lines[i+1]:
                        date_str = line.split("Date:")[-1].strip()
                        time_str = lines[i+1].split("Time:")[-1].strip()
                        timestamp = f"{date_str} {time_str}"
                        timestamp = datetime.datetime.strptime(timestamp, "%d/%m/%Y %H:%M UTC")
                    if line.startswith("Number of Angles:"):
                        num_angles = int(line.split("Number of Angles:")[-1].strip())
                    if line.startswith("Sonar Distance (mm):"):
                        sweep.sonar_distance = int(line.split("Sonar Distance (mm):")[-1].strip())
                    if line.startswith("Recorded Angles (degrees):"):
                        angles = list(map(int, re.findall(r"-?\d+", line)))  
                    if line.startswith("Raw Measurements"):
                        angles_added = set()
                        for j in range(i+1, len(lines)):
                            if lines[j].startswith("Angle (degrees):"):
                                angle_val = int(lines[j].split("Angle (degrees):")[-1].strip())
                                if angle_val < -90 or angle_val > 90:
                                    continue
                                # Skip if angle 0 already added
                                if angle_val == 0 and 0 in angles_added:
                                    continue
                                meas = LidarMeasurement()
                                meas.timestamp = timestamp
                                meas.sonar_distance = sweep.sonar_distance
                                meas.angle = angle_val
                                distances_line = lines[j + 1]
                                distances = list(map(int, distances_line.split(":")[-1].strip().split(", ")))
                                meas.distances = distances
                                num_distances += len(distances)
                                valid_distances = [d for d in distances if d > 500 and d < 5000]
                                if valid_distances:
                                    meas.avg_distance = sum(valid_distances) / len(valid_distances)
                                sweep.measurements.append(meas)
                                angles_added.add(angle_val)
                all_sweeps.append(sweep) 

        except FileNotFoundError:
            print(f"Error: File {file_name} not found.")

    calculate_coordinates(all_sweeps)

    combined_by_angle = {}
    for sweep in all_sweeps:
        for meas in sweep.measurements:
            angle = meas.angle
            if angle not in combined_by_angle:
                combined_by_angle[angle] = []
            combined_by_angle[angle].append(meas)

    unique_sonar_distances = set()
    all_timestamps = []
    for sweep in all_sweeps:
        if hasattr(sweep, "sonar_distance"):
            unique_sonar_distances.add(sweep.sonar_distance)
        if hasattr(sweep, "sonar_distances"):
            unique_sonar_distances.update(sweep.sonar_distances)
        # Collect all timestamps from measurements, avoiding duplicates
        for meas in getattr(sweep, "measurements", []):
            if hasattr(meas, "timestamp") and meas.timestamp is not None:
                if meas.timestamp not in all_timestamps:
                    all_timestamps.append(meas.timestamp)
    unique_sonar_distances = {d for d in unique_sonar_distances if d}

    combined_sweep = LidarSweep()
    combined_sweep.sonar_distances = list(unique_sonar_distances)
    combined_sweep.timestamps = all_timestamps
    for angle, measurements in combined_by_angle.items():
        combined_meas = LidarMeasurement()
        combined_meas.angle = angle

        for m in measurements:
            if hasattr(m, "distances") and m.distances:
                combined_meas.distances.extend(m.distances)
            if hasattr(m, "raw_points") and m.raw_points:
                combined_meas.raw_points.extend(m.raw_points)
            if hasattr(m, "in_water"):
                if m.in_water is True:
                    combined_meas.in_water = True

        combined_sweep.measurements.append(combined_meas)

    kde_centroids_by_angle(combined_sweep)
    find_banks(combined_sweep)
    process_kde_centroids(combined_sweep)
    calculate_cross_sectional_area(combined_sweep)

    def print_combined_sweep_summary(sweeps):
        """
        Prints and saves a summary of all sweeps for the day.
        """
        summary_lines = []
        summary_lines.append("=== Sweep Summary ===")
        all_timestamps = []
        all_sonar_distances = set()
        total_measurements = 0
        total_distances = 0
        total_valid = 0
        total_errors = 0
        total_timeouts = 0
        total_certain_centroids = 0
        total_uncertain_centroids = 0
        total_raw_points = 0

        for sweep in sweeps:
            timestamps = getattr(sweep, 'timestamps', [])
            all_timestamps.extend(timestamps)
            sonar_distances = getattr(sweep, 'sonar_distances', [])
            all_sonar_distances.update(sonar_distances)
            measurements = getattr(sweep, 'measurements', [])
            total_measurements += len(measurements)
            for meas in measurements:
                if hasattr(meas, "distances") and isinstance(meas.distances, list):
                    distances = meas.distances
                    total_distances += len(distances)
                    total_errors += sum(1 for d in distances if d > 9990 and d != 9999)
                    total_timeouts += sum(1 for d in distances if d == 9999)
                    total_valid += sum(1 for d in distances if d < 9990)
                if hasattr(meas, "raw_points") and isinstance(meas.raw_points, list):
                    total_raw_points += len(meas.raw_points)
                if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None:
                    total_certain_centroids += 1
                if hasattr(meas, "uncertain_centroid") and meas.uncertain_centroid is not None:
                    total_uncertain_centroids += 1

        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            summary_lines.append(f"Timestamps: {all_timestamps[0]} to {all_timestamps[-1]}")
            # Use the date of the first timestamp for the filename
            date_str = all_timestamps[0].strftime("%Y%m%d")
        else:
            summary_lines.append("Timestamps: None")
            date_str = "unknown"
        summary_lines.append(f"Sonar Distances: {sorted(all_sonar_distances)} (count: {len(all_sonar_distances)})")
        summary_lines.append(f"Total Sweeps: {len(sweeps)}")
        summary_lines.append(f"Total Measurements: {total_measurements}")
        summary_lines.append(f"Total distances: {total_distances}")
        summary_lines.append(f"  Valid: {total_valid}")
        summary_lines.append(f"  Errors: {total_errors}")
        summary_lines.append(f"  Timeouts: {total_timeouts}")
        summary_lines.append(f"Total certain centroids: {total_certain_centroids}")
        summary_lines.append(f"Total uncertain centroids: {total_uncertain_centroids}")
        summary_lines.append(f"Total raw points: {total_raw_points}")
        summary_lines.append("=== End of Summary ===\n")

        # Print to console
        print("\n".join(summary_lines))

        # Save to file with date in filename
        out_dir = os.path.join(PROJECT_ROOT, "data", today) 
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}_summary.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(summary_lines))

    # Call the summary function after processing
    print_combined_sweep_summary([combined_sweep])
    graph([combined_sweep])

if __name__ == "__main__":

    execute()
