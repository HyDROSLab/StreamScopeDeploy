"""
StreamScope Cross-Section Analysis Script

This script processes data from StreamScope sweeps to analyze stream cross-sections.
It reads measurement logs, computes coordinates for points, removes outliers, clusters points to find centroids,
identifies stream banks, and visualizes the cross-section using matplotlib.

Classes:
    LidarPoint: Represents a single coordinate point in 2D space.
    LidarMeasurement: Holds all measurements for a single angle sweep.
    LidarSweep: Represents a full sweep of laser measurements at a given timestamp.

Functions:
    calculate_coordinates(sweeps): Computes (x, y) coordinates for all measurements, accounting for refraction.
    remove_outliers(sweep, eps=50, min_samples=3): Removes outlier points from measurements using DBSCAN clustering.
    process_centroids(sweep): Clusters inlier points to find centroids for each angle.
    find_banks(sweep): Identifies the left and right stream banks based on centroids and water detection.
    graph(sweeps): Visualizes the processed data, including raw points, centroids, and banks.
    execute(): Main data processing and visualization pipeline.
"""

import os
import re
import matplotlib.pyplot as plt
import datetime
import glob
import math
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import pairwise_distances
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, label, find_objects
import os

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
        return hash((round(self.x, 6), round(self.y, 6)))

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

        # --- Plot for this measurement ---
        # plt.figure(figsize=(6, 6))
        # plt.scatter(coords[:, 0], coords[:, 1], s=10, color='black', alpha=0.5, label='Raw Points')
        # plt.contourf(xx, yy, np.exp(log_density_2d), levels=20, cmap='Blues', alpha=0.5)
        # plt.scatter([x_certain], [y_certain], color='green', marker='X', s=100, label='Certain Centroid')
        # if uncertain_centroid is not None:
            # plt.scatter([uncertain_centroid.x], [uncertain_centroid.y], color='orange', marker='P', s=100, label='Uncertain Centroid')
        # plt.title(f"Angle {meas.angle}° KDE & Centroids")
        # plt.xlabel("X (mm)")
        # plt.ylabel("Y (mm)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        # Wait for user to close the plot before continuing
        # plt.close()

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
                threshold = max(0, min(400, avg_dist_to_sonar))  # Clamp between 50 and 400 mm
            else:
                threshold = 200  # Fallback default

            # if abs(abs(cc.y) - sonar_dist) <= threshold and abs(meas.angle) <= 15:
                # meas.certain_centroid = None
            


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
        # Find a valid sonar distance for fallback
        valid_sonar = None
        for meas in sweep.measurements:
            if hasattr(meas, "sonar_distance") and 500 < meas.sonar_distance < 9000:
                valid_sonar = meas.sonar_distance
                break
        for meas in sweep.measurements:
            # Use valid sonar distance if current is invalid
            water_level = meas.sonar_distance
            if not (500 < water_level < 9000):
                water_level = valid_sonar if valid_sonar is not None else 0
            rad_angle = math.radians(meas.angle)
            for i, dist in enumerate(meas.distances):
                if not (9000 <= dist <= 10000) and water_level > 0:
                    da = water_level / math.cos(rad_angle) if water_level else 0
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
                elif 500 < dist < 15000 and not (9000 <= dist <= 10000) and water_level <= 0:
                    xa = math.sin(rad_angle) * dist
                    ya = math.cos(rad_angle) * dist
                    point = LidarPoint(-1 * xa, -1 * ya)
                    meas.raw_points.append(point)

def remove_outliers(sweep, eps=50, min_samples=3):

    """
    Removes outlier points from each measurement in the sweep using DBSCAN clustering.

    Args:
        sweep (LidarSweep): The sweep to process.
        eps (float): DBSCAN epsilon parameter.
        min_samples (int): Minimum samples for DBSCAN.
    """

    for measurement in sweep.measurements:
        if len(measurement.raw_points) < 2:
            continue

        coords = np.array([point.as_tuple() for point in measurement.raw_points])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = dbscan.labels_

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            continue

        largest_cluster_label = max(unique_labels, key=list(labels).count)
        inlier_tuples = [coords[i] for i in range(len(labels)) if labels[i] == largest_cluster_label]
        outlier_tuples = [coords[i] for i in range(len(labels)) if labels[i] == -1]
        measurement.inliers = [LidarPoint(x, y) for x, y in inlier_tuples]
        measurement.outliers = [LidarPoint(x, y) for x, y in outlier_tuples]

def process_centroids(sweep):
    """
    Clusters inlier points for each measurement to find certain and uncertain centroids.

    Args:
        sweep (LidarSweep): The sweep to process.
    """

    angle_to_points = {meas.angle: list(meas.raw_points) for meas in sweep.measurements}

    missing_centroids = []

    for meas in sweep.measurements:
        sonar_dist = getattr(meas, "sonar_distance", 0)
        if hasattr(meas, "inliers"):
            if meas.in_water:
                meas.inliers = [pt for pt in meas.inliers if meas.in_water]
            elif any(m.in_water and m.angle == meas.angle for m in sweep.measurements):
                meas.inliers = []

        if hasattr(meas, "inliers") and len(meas.inliers) >= 10:
            coords = np.array([pt.as_tuple() for pt in meas.inliers])
            k = min(3, len(coords))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(coords)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Convert centroids to LidarPoint objects
            centroid_points = [LidarPoint(c[0], c[1]) for c in centroids]

            # Find the centroid with the highest density (most points assigned)
            label_counts = np.bincount(labels)
            densest_idx = np.argmax(label_counts)
            certain_centroid = centroid_points[densest_idx]

            # For uncertain centroid, pick the farthest centroid from origin (or sonar_dist)
            farthest_idx = np.argmax([abs(pt.y) for pt in centroid_points])
            uncertain_candidate = centroid_points[farthest_idx]
            if abs(uncertain_candidate.y - 50) > sonar_dist:
                uncertain_centroid = uncertain_candidate
            else:
                uncertain_centroid = None

            meas.certain_centroid = certain_centroid
            meas.uncertain_centroid = uncertain_centroid

            # If only one centroid, fallback to farthest inlier or raw point for uncertain
            if len(centroid_points) == 1:
                farthest_pt = max(
                    (pt for pt in meas.inliers if abs(pt.y - 50) > sonar_dist),
                    key=lambda pt: abs(pt.y),
                    default=None
                )
                if farthest_pt:
                    meas.uncertain_centroid = farthest_pt
                else:
                    farthest_pt = max((pt for pt in meas.raw_points if abs(pt.y - 50) > sonar_dist), key=lambda pt: abs(pt.y), default=None)
                    meas.uncertain_centroid = farthest_pt if farthest_pt else None
        else:
            farthest_pt = max(
                (pt for pt in meas.raw_points if abs(pt.y - 50) > sonar_dist),
                key=lambda pt: abs(pt.y),
                default=None
            )
            meas.uncertain_centroid = farthest_pt if farthest_pt else None
            meas.certain_centroid = None

        # Ensure uncertain centroid does not exist at y values greater than sonar distance
        if meas.uncertain_centroid is not None and abs(meas.uncertain_centroid.y) <= sonar_dist:
            meas.uncertain_centroid = None

        if meas.certain_centroid is None or meas.uncertain_centroid is None:
            missing_centroids.append(meas.angle)

    if missing_centroids:
        for meas in sweep.measurements:
            sonar_dist = getattr(meas, "sonar_distance", 0)
            if meas.angle in missing_centroids and meas.raw_points:
                furthest_raw = max(
                    (pt for pt in meas.raw_points if abs(pt.y - 50) > sonar_dist),
                    key=lambda pt: abs(pt.y),
                    default=None
                )
                if furthest_raw and abs(furthest_raw.y - 50) > sonar_dist:
                    meas.uncertain_centroid = furthest_raw
                    meas.certain_centroid = None
                else:
                    meas.uncertain_centroid = None
                    meas.certain_centroid = None

def find_banks(sweep):

    # Find left and right bank x-coordinates based on sonar distance

    sorted_measurements = sorted(sweep.measurements, key=lambda m: m.angle)

    left_bank_x = None
    right_bank_x = None

    sonar_dist = sweep.sonar_distances[0] if hasattr(sweep, "sonar_distances") and sweep.sonar_distances else None
    if sonar_dist is None:
        sweep.left_bank = None
        sweep.right_bank = None
    else:
        # Find left bank: minimum angle with a centroid whose y is close to sonar distance
        for m in sorted_measurements:
            cc = getattr(m, "certain_centroid", None)
            uc = getattr(m, "uncertain_centroid", None)
            pt = cc if cc is not None else uc
            if pt is not None and abs(abs(pt.y) - abs(sonar_dist)) < 100:
                left_bank_x = pt.x
                break

        # Find right bank: maximum angle with a centroid whose y is close to sonar distance
        for m in reversed(sorted_measurements):
            cc = getattr(m, "certain_centroid", None)
            uc = getattr(m, "uncertain_centroid", None)
            pt = cc if cc is not None else uc
            if pt is not None and abs(abs(pt.y) - abs(sonar_dist)) < 100:
                right_bank_x = pt.x
                break

        sweep.left_bank = LidarPoint(left_bank_x, sonar_dist) if left_bank_x is not None else None
        sweep.right_bank = LidarPoint(right_bank_x, sonar_dist) if right_bank_x is not None else None

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

    Args:
        sweeps (list): List of LidarSweep objects to visualize.
    """

    plt.figure(figsize=(8, 8))

    mm_to_ft = 1 / 304.8

    raw_x = []
    raw_y = []
    centroid_x = []
    centroid_y = []
    uncertain_centroid_x = []
    uncertain_centroid_y = []

    for sweep in sweeps:
        for measurement in sweep.measurements:
            for point in getattr(measurement, "raw_points", []):
                raw_x.append(point.x * mm_to_ft)
                raw_y.append(-point.y * mm_to_ft)
            if hasattr(measurement, "certain_centroid") and measurement.certain_centroid is not None:
                centroid_x.append(measurement.certain_centroid.x * mm_to_ft)
                centroid_y.append(-measurement.certain_centroid.y * mm_to_ft)
            if hasattr(measurement, "uncertain_centroid") and measurement.uncertain_centroid is not None:
                uncertain_centroid_x.append(measurement.uncertain_centroid.x * mm_to_ft)
                uncertain_centroid_y.append(-measurement.uncertain_centroid.y * mm_to_ft)

    left_bank = getattr(sweep, "left_bank", None)
    right_bank = getattr(sweep, "right_bank", None)

    left_bank_plot_x = left_bank.x * mm_to_ft if left_bank else None
    right_bank_plot_x = right_bank.x * mm_to_ft if right_bank else None

    # Only plot bank lines if both left and right bank x values are available
    if left_bank_plot_x is not None and right_bank_plot_x is not None:
        pass
        # plt.axvline(x=right_bank_plot_x, color='purple', linestyle='-.', linewidth=2, label="Left Bank")
        # plt.axvline(x=left_bank_plot_x, color='brown', linestyle='-.', linewidth=2, label="Right Bank")

    sonar_distances = [d for sweep in sweeps if hasattr(sweep, "sonar_distances") for d in sweep.sonar_distances]
    if sonar_distances:
        min_sonar = min(sonar_distances) * mm_to_ft
        max_sonar = max(sonar_distances) * mm_to_ft
        plt.axhspan(max_sonar, min_sonar, color='red', alpha=0.50, label="Sonar Distance Range")

    if raw_x and raw_y:
        plt.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")

    # Plot certain centroids as scatter
    if centroid_x and centroid_y:
        pass
        # plt.scatter(centroid_x, centroid_y, s=80, color='green', marker='X', label="Certain Centroids", zorder=90, edgecolors='black', linewidths=1.2)

    # Plot contour for certain centroids (sorted by angle)
    angle_centroid_pairs = []
    for sweep in sweeps:
        for measurement in sweep.measurements:
            if (
                hasattr(measurement, "certain_centroid")
                and measurement.certain_centroid is not None
            ):
                angle_centroid_pairs.append((measurement.angle, measurement.certain_centroid))
    angle_centroid_pairs.sort(key=lambda x: x[0])
    if angle_centroid_pairs:
        cx = [(c.x * mm_to_ft) for _, c in angle_centroid_pairs]
        cy = [-(c.y * mm_to_ft) for _, c in angle_centroid_pairs]
        # plt.plot(cx, cy, color='lime', linewidth=2, zorder=1, label="Certain Centroids Contour")

    # Plot uncertain centroids as scatter (optional, uncomment if needed)
    if uncertain_centroid_x and uncertain_centroid_y:
        pass
        # plt.scatter(uncertain_centroid_x, uncertain_centroid_y, color='orange', marker='X', s=50, edgecolors='black', label="Uncertain Centroids")

    # Plot the area polygon, ensuring correct order and sign
    # avg_x = [x * mm_to_ft for x in sweep.area_polygons["avg"][0]]
    # avg_y = [y * mm_to_ft for y in sweep.area_polygons["avg"][1]]
    # if len(avg_x) == len(avg_y) and len(avg_x) > 0:
        # poly_points = list(zip(avg_x, avg_y))
        # poly_points_sorted = sorted(poly_points, key=lambda p: p[0])
        # poly_x_sorted = [x for x, y in poly_points_sorted]
        # poly_y_sorted = [-y for x, y in poly_points_sorted]
        # plt.fill(poly_x_sorted, poly_y_sorted, color='green', alpha=0.3, label=f"Avg Area ({sweep.area:.1f} m²)")

    # --- Overlay Falls_Creek_Manual_Survey CSV ---

    csv_path = os.path.join(os.path.dirname(__file__), "Falls_Creek_Manual_Survey.csv")
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

        xs_centered = (xs_mm - x0 - 1010) * mm_to_ft
        ys_centered = (ys_mm - y0 + 3025) * mm_to_ft

        # plt.scatter(-xs_centered, ys_centered, color='blue', s=80, marker='X', edgecolors='black', linewidths=1.2, label="Manual Survey Points", zorder=50)

        sorted_indices = np.argsort(xs_centered)
        # plt.plot(-xs_centered[sorted_indices], ys_centered[sorted_indices], color='blue', linewidth=2, label="Manual Survey Contour", zorder=51)
    else:
        print(f"Manual survey CSV not found at {csv_path}")

    # Determine date and time range from sweeps
    timestamps = getattr(sweep, "timestamps", [])
    if timestamps:
        timestamps = sorted(timestamps)
        date_str = timestamps[0].strftime("%Y-%m-%d")
        start_time = timestamps[0].strftime("%H:%M")
        end_time = timestamps[-1].strftime("%H:%M")
        plt.title(f"StreamScope Cross-Section - Plum Creek ({date_str}, {start_time}–{end_time} UTC)")
    else:
        plt.title("StreamScope Cross-Section")
    plt.xlabel("X (ft)")
    plt.ylabel("Y (ft)")

    # plt.xlim(-20, 20) # Sapello XLIM
    # plt.ylim(20, 0) # Sapello YLIM
    plt.xlim(-25, 25)
    plt.ylim(30, 0)
    plt.xticks(np.arange(-25, 26, 0.50))
    plt.yticks(np.arange(0, 31, 0.50))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # plt.gca().set_aspect('equal', adjustable='box')
    main_legend = plt.legend(loc='lower left')
    plt.gca().add_artist(main_legend)

    # Error, Valid, and Timeout Inset Plot
    total_points = 0
    valid_points = 0
    error_points = 0
    timeout_points = 0
    for sweep in sweeps:
        for measurement in sweep.measurements:
            for d in getattr(measurement, "distances", []):
                total_points += 1
                if d == 9999:
                    timeout_points += 1
                elif d > 9990:
                    error_points += 1
                else:
                    valid_points += 1

    if total_points > 0:
        valid_pct = valid_points / total_points * 100
        error_pct = error_points / total_points * 100
        timeout_pct = timeout_points / total_points * 100
    else:
        valid_pct = error_pct = timeout_pct = 0

    sizes = [valid_pct, error_pct, timeout_pct]
    counts = [valid_points, error_points, timeout_points]
    labels = ['Valid', 'Error', 'Timeout']
    colors = ['#4CAF50', '#FF9800', '#F44336']

    def make_autopct(sizes, counts):
        def my_autopct(pct):
            total = sum(counts)
            count = int(round(pct * total / 100.0))
            idx = int(round(pct * len(counts) / 100.0))
            my_autopct.counter += 1
            i = my_autopct.counter - 1
            if i < len(counts):
                return f"{pct:.1f}%\n({counts[i]})"
            else:
                return f"{pct:.1f}%"
        my_autopct.counter = 0
        return my_autopct
    # wedges, texts, autotexts = axins.pie(
        # sizes, labels=labels, colors=colors,
        # autopct=make_autopct(sizes, counts), startangle=90, textprops={'fontsize': 9}
    # )
    # axins.set_title("Point Type %", fontsize=10)
    plt.show()

def execute(folder):
    """
    Main data processing and visualization pipeline.
    Reads log files, processes sweeps, computes coordinates, removes outliers,
    finds centroids and banks, and visualizes the results.
    """

    file_pattern = f"/home/braden/StreamScopeDeploy/data/plumcreek/{folder}/streamscope_log_*.txt"
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
                                # Set sonar_distance for measurement
                                if 0 < sweep.sonar_distance < 9000:
                                    meas.sonar_distance = sweep.sonar_distance
                                else:
                                    meas.sonar_distance = 0
                                meas.angle = angle_val
                                distances_line = lines[j + 1]
                                distances = list(map(int, distances_line.split(":")[-1].strip().split(", ")))
                                meas.distances = distances
                                num_distances += len(distances)
                                valid_distances = [d for d in distances if d > 500 and d < 15000]
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

    def print_combined_sweep_summary(sweep):
        print("=== Combined Sweep Summary ===")
        timestamps = [getattr(sweep, 'timestamps', None)]
        if timestamps:
            timestamps = sorted(timestamps)
            print(f"Timestamps: {timestamps[0]} to {timestamps[-1]}")
        else:
            print("Timestamps: None")
        print(f"Accelerometer Available: {getattr(sweep, 'accelerometer_available', None)}")
        print(f"Sonar Distances: {getattr(sweep, 'sonar_distances', None)} (length: {len(getattr(sweep, 'sonar_distances', []))})")
        print(f"Area: {getattr(sweep, 'area', None)}")
        print(f"Stream Width: {getattr(sweep, 'stream_width', None)}")
        print(f"Left Bank: {sweep.left_bank.x if sweep.left_bank else None}")
        print(f"Right Bank: {sweep.right_bank.x if sweep.right_bank else None}")
        print(f"Number of Measurements: {len(getattr(sweep, 'measurements', []))}")
        print()

        total_distances = 0
        total_valid = 0
        total_errors = 0
        total_timeouts = 0
        total_certain_centroids = 0
        total_uncertain_centroids = 0
        total_raw_points = 0

        for meas in getattr(sweep, 'measurements', []):
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

        print(f"Total distances: {total_distances}")
        print(f"  Valid: {total_valid}")
        print(f"  Errors: {total_errors}")
        print(f"  Timeouts: {total_timeouts}")
        print(f"Total certain centroids: {total_certain_centroids}")
        print(f"Total uncertain centroids: {total_uncertain_centroids}")
        print(f"Total raw points: {total_raw_points}")
        print("=== End of Summary ===\n")

    print_combined_sweep_summary(combined_sweep)
    graph([combined_sweep])

if __name__ == "__main__":
    while True:
        print("\nStreamScope Cross-Section.\n")
        print("1. Cross-Section View.")
        print("2. Exit.")
        choice = input("Select an option: ") 
        
        if choice == '1':
            date_input = input("Enter date (MMDDYYYY): ").strip()
            base_dir = "/home/braden/StreamScopeDeploy/data/plumcreek/"
            folder_pattern = os.path.join(base_dir, f"{date_input}")
            if not os.path.isdir(folder_pattern):
                print(f"Folder for date {date_input} not found at {folder_pattern}. Please try again.\n")
                continue
            else:
                execute(date_input)
        elif choice == '2':
            print("Exiting the program.")
            break
