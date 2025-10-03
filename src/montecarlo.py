"""
Monte Carlo view factor calculations.

This module implements view factor calculations using Monte Carlo
ray tracing with statistical sampling and uncertainty estimation.
"""

import time
import numpy as np
from typing import Tuple
import logging

from .constants import EPS, STATUS_CONVERGED, STATUS_REACHED_LIMITS, STATUS_FAILED
from .geometry import Rectangle, ViewFactorResult, validate_geometry

logger = logging.getLogger(__name__)


class MonteCarloCalculator:
    """Calculator using Monte Carlo ray tracing method.
    
    Implements view factor calculation using random ray sampling
    with statistical uncertainty estimation.
    """
    
    def __init__(self, 
                 n_samples: int = 200000,
                 target_rel_ci: float = 0.02,
                 max_batches: int = 50,
                 seed: int = 42) -> None:
        """Initialise Monte Carlo calculator.
        
        Args:
            n_samples: Total number of ray samples
            target_rel_ci: Target relative half-width of 95% CI
            max_batches: Maximum number of batches
            seed: Random number generator seed
        """
        self._n_samples = n_samples
        self._target_rel_ci = target_rel_ci
        self._max_batches = max_batches
        self._seed = seed
        self._name = "montecarlo"
        
        # Set random seed for reproducibility
        np.random.seed(seed)
    
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult:
        """Calculate view factor using Monte Carlo method.
        
        Args:
            emitter: Source surface geometry
            receiver: Target surface geometry
            
        Returns:
            ViewFactorResult with calculated value and metadata
        """
        start_time = time.perf_counter()
        
        # Validate geometry
        validate_geometry(emitter, receiver)
        
        try:
            # Run Monte Carlo simulation
            view_factor, uncertainty, converged, samples_used = self._monte_carlo_simulation(
                emitter, receiver
            )
            
            computation_time = time.perf_counter() - start_time
            
            return ViewFactorResult(
                value=view_factor,
                uncertainty=uncertainty,
                converged=converged,
                iterations=samples_used,
                computation_time=computation_time,
                method_used=self._name
            )
            
        except Exception as e:
            logger.error(f"Monte Carlo calculation failed: {e}")
            raise
    
    def _monte_carlo_simulation(self, 
                              emitter: Rectangle, 
                              receiver: Rectangle) -> Tuple[float, float, bool, int]:
        """Run Monte Carlo simulation with adaptive sampling.
        
        Args:
            emitter: Source rectangle
            receiver: Target rectangle
            
        Returns:
            Tuple of (view_factor, uncertainty, converged, samples_used)
        """
        # Pre-compute geometry
        n1 = emitter.normal
        n2 = receiver.normal
        
        # Auto-orient normals to face each other
        centre_direction = receiver.centroid - emitter.centroid
        if np.dot(n1, centre_direction) < 0:
            n1 = -n1
        if np.dot(n2, -centre_direction) < 0:
            n2 = -n2
        
        # Batch processing for convergence monitoring
        batch_size = max(self._n_samples // self._max_batches, 1000)
        total_samples = 0
        sum_contributions = 0.0
        sum_squared_contributions = 0.0
        
        converged = False
        
        for batch in range(self._max_batches):
            # Generate random samples for this batch
            batch_contributions = self._sample_batch(
                batch_size, emitter, receiver, n1, n2
            )
            
            # Update statistics
            total_samples += len(batch_contributions)
            sum_contributions += np.sum(batch_contributions)
            sum_squared_contributions += np.sum(batch_contributions ** 2)
            
            # Check convergence if we have enough samples
            if total_samples >= 10000:
                mean_estimate = sum_contributions / total_samples
                variance = (sum_squared_contributions / total_samples - mean_estimate ** 2)
                std_error = np.sqrt(variance / total_samples)
                
                # 95% confidence interval half-width
                ci_half_width = 1.96 * std_error
                
                # Check relative convergence
                if mean_estimate > 0:
                    relative_ci = ci_half_width / mean_estimate
                    if relative_ci < self._target_rel_ci:
                        converged = True
                        break
            
            # Stop if we've used all requested samples
            if total_samples >= self._n_samples:
                break
        
        # Calculate final statistics
        if total_samples > 0:
            view_factor = sum_contributions / total_samples
            variance = max(0, sum_squared_contributions / total_samples - view_factor ** 2)
            uncertainty = 1.96 * np.sqrt(variance / total_samples)  # 95% CI half-width
        else:
            view_factor = 0.0
            uncertainty = 1.0
        
        return view_factor, uncertainty, converged, total_samples
    
    def _sample_batch(self, 
                     batch_size: int,
                     emitter: Rectangle,
                     receiver: Rectangle,
                     n1: np.ndarray,
                     n2: np.ndarray) -> np.ndarray:
        """Generate a batch of Monte Carlo samples.
        
        Args:
            batch_size: Number of samples in this batch
            emitter: Source rectangle
            receiver: Target rectangle
            n1: Emitter normal vector
            n2: Receiver normal vector
            
        Returns:
            Array of view factor contributions for each sample
        """
        contributions = np.zeros(batch_size)
        
        for i in range(batch_size):
            # Random point on emitter surface
            s1 = np.random.random()
            t1 = np.random.random()
            p1 = emitter.origin + s1 * emitter.u_vector + t1 * emitter.v_vector
            
            # Random point on receiver surface
            s2 = np.random.random()
            t2 = np.random.random()
            p2 = receiver.origin + s2 * receiver.u_vector + t2 * receiver.v_vector
            
            # Calculate view factor contribution
            r_vec = p2 - p1
            r = np.linalg.norm(r_vec)
            
            if r > EPS:  # Avoid division by zero
                r_hat = r_vec / r
                
                cos1 = np.dot(n1, r_hat)
                cos2 = -np.dot(n2, r_hat)
                
                if cos1 > 0 and cos2 > 0:
                    # View factor kernel
                    kernel = (cos1 * cos2) / (np.pi * r * r)
                    
                    # Monte Carlo contribution (multiply by receiver area)
                    contributions[i] = kernel * receiver.area
        
        return contributions
    
    def _hemisphere_sampling(self, 
                           source_point: np.ndarray,
                           source_normal: np.ndarray,
                           n_rays: int) -> Tuple[np.ndarray, float]:
        """Alternative hemisphere sampling method.
        
        Args:
            source_point: Point on source surface
            source_normal: Normal vector at source point
            n_rays: Number of rays to sample
            
        Returns:
            Tuple of (ray_directions, solid_angle_weight)
        """
        # Generate random directions in hemisphere
        # Using cosine-weighted sampling for better efficiency
        
        # Random numbers
        u1 = np.random.random(n_rays)
        u2 = np.random.random(n_rays)
        
        # Cosine-weighted hemisphere sampling
        cos_theta = np.sqrt(u1)
        sin_theta = np.sqrt(1 - u1)
        phi = 2 * np.pi * u2
        
        # Local coordinates (z = normal direction)
        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta
        
        # Create orthonormal basis with normal as z-axis
        n = source_normal / np.linalg.norm(source_normal)
        
        # Choose arbitrary tangent vector
        if abs(n[0]) < 0.9:
            t1 = np.array([1.0, 0.0, 0.0])
        else:
            t1 = np.array([0.0, 1.0, 0.0])
        
        t1 = t1 - np.dot(t1, n) * n
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        
        # Transform to world coordinates
        directions = np.zeros((n_rays, 3))
        for i in range(n_rays):
            directions[i] = x[i] * t1 + y[i] * t2 + z[i] * n
        
        # Weight for cosine-weighted sampling is 1/π
        weight = 1.0 / np.pi
        
        return directions, weight
    
    def estimate_samples_needed(self, 
                              emitter: Rectangle, 
                              receiver: Rectangle,
                              target_accuracy: float = 0.01) -> int:
        """Estimate number of samples needed for target accuracy.
        
        Args:
            emitter: Source rectangle
            receiver: Target rectangle
            target_accuracy: Target relative accuracy (e.g., 0.01 for 1%)
            
        Returns:
            Estimated number of samples needed
        """
        # Run a small pilot study
        pilot_samples = 10000
        
        # Temporarily save current seed and set deterministic seed
        current_seed = np.random.get_state()
        np.random.seed(12345)
        
        try:
            # Get pilot estimate
            pilot_result, pilot_uncertainty, _, _ = self._monte_carlo_simulation(
                emitter, receiver
            )
            
            if pilot_result > 0 and pilot_uncertainty > 0:
                # Estimate variance
                pilot_variance = (pilot_uncertainty / 1.96) ** 2 * pilot_samples
                
                # Samples needed for target accuracy
                target_variance = (target_accuracy * pilot_result) ** 2
                samples_needed = int(pilot_variance / target_variance)
                
                return max(samples_needed, 10000)  # Minimum 10k samples
            else:
                return self._n_samples  # Use default if pilot fails
                
        finally:
            # Restore random state
            np.random.set_state(current_seed)
