from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

from .stages import build_templates
from ..types import Signal, References

def identify_components(
    components: np.ndarray, 
    wavelength: np.ndarray,
    references: References,
    top_n: int = 3,
    species_filter: list[str] | None = None,
) -> list[list[tuple[str, float]]]:
    """
    Match extracted spectral components to known reference elements.
    
    Args:
        components: Array of shape (n_components, n_wavelengths)
        wavelength: Array of shape (n_wavelengths,)
        references: The Loaded reference lines
        top_n: Number of top matches to return per component
        species_filter: Optional list of species to consider (filters out noise-prone 
                       elements like actinides that have too many lines)
        
    Returns:
        A list of lists, where each inner list contains (Element, Score) tuples
        for a component.
    """
    n_comps = components.shape[0]
    results = []
    
    # Build templates for all references once to check against
    # We use a standard FWHM for matching
    # Create a dummy signal to build templates
    dummy_sig = Signal(wavelength=wavelength, intensity=np.zeros_like(wavelength))
    # This is expensive if we do it every time, but safe for now.
    # We assume the components are already continuum removed or similar to emission lines.
    # However, PCA/ICA components can be negative or mixed. 
    # We only look at positive similarity for emission lines.
    
    # We can reuse the build_templates logic but we need to access the raw matrix
    # Let's build templates for all available species
    templates = build_templates(dummy_sig, references, fwhm_nm=0.5, species_filter=species_filter)
    ref_matrix = templates.matrix
    ref_species = templates.species
    
    # Normalize reference matrix columns
    ref_norms = np.linalg.norm(ref_matrix, axis=0)
    ref_norms[ref_norms == 0] = 1.0
    ref_matrix_norm = ref_matrix / ref_norms
    
    for i in range(n_comps):
        comp = components[i]
        # Ensure component is positive (some methods return negative spectra)
        # If it's mostly negative, flip it? 
        # For PCA/ICA, sign is ambiguous. For MCR, it's non-negative.
        # Heuristic: if max abs value is negative, flip.
        if np.max(comp) < np.abs(np.min(comp)):
             comp = -comp
             
        # Zero out negative values for matching against emission lines
        comp_pos = np.maximum(comp, 0)
        comp_norm = np.linalg.norm(comp_pos)
        if comp_norm > 0:
            comp_pos /= comp_norm
            
        # Calculate dot product (cosine similarity) with all references
        scores = comp_pos @ ref_matrix_norm
        
        # Get top N matches
        top_indices = np.argsort(scores)[::-1][:top_n]
        matches = [(ref_species[idx], float(scores[idx])) for idx in top_indices]
        results.append(matches)
        
    return results



def _stack_signals(signals: Sequence[Signal]) -> np.ndarray:
    """Stack signal intensities into a matrix (N_samples x M_wavelengths)."""
    if not signals:
        raise ValueError("No signals provided")
    # Check if all signals have the same length
    n_points = signals[0].intensity.size
    if any(s.intensity.size != n_points for s in signals):
        raise ValueError("All signals must have the same number of points for multivariate analysis. Run resample() first.")
    
    return np.stack([s.intensity for s in signals])


def analyze_pca(signals: Sequence[Signal], n_components: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis (PCA) on a set of signals.

    Args:
        signals: List of Signal objects (must be aligned/resampled to same grid).
        n_components: Number of principal components to compute.

    Returns:
        components: The "Score" of each sample (shape: N_samples x n_components)
        pca_components: The "Spectrum" of each PC (shape: n_components x M_wavelengths)
    """
    X = _stack_signals(signals)
    
    # Standardize (crucial for PCA)
    X_std = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_std)
    
    return components, pca.components_


def analyze_ica(signals: Sequence[Signal], n_components: int = 10) -> np.ndarray:
    """
    Perform Independent Component Analysis (ICA) on a set of signals.

    Args:
        signals: List of Signal objects.
        n_components: Number of components to extract.

    Returns:
        sources: The "Pure Spectra" found (shape: M_wavelengths x n_components)
    """
    X = _stack_signals(signals)
    # ICA often works better without standardization or with just centering
    X_centered = X - X.mean(axis=0)
    
    ica = FastICA(n_components=n_components, random_state=42)
    # Transpose X to treat wavelengths as samples if we want spectral sources?
    # The user's snippet was: sources = ica.fit_transform(X_centered.T)
    # If X is (N_samples, M_wavelengths):
    # X.T is (M_wavelengths, N_samples).
    # ICA fit_transform(X.T) -> (M_wavelengths, n_components).
    # The columns of 'sources' are the independent spectral components.
    sources = ica.fit_transform(X_centered.T)
    
    return sources


def analyze_mcr(signals: Sequence[Signal], n_components: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Multivariate Curve Resolution (MCR-ALS) on a set of signals.

    Args:
        signals: List of Signal objects.
        n_components: Number of components to resolve.

    Returns:
        concentrations: The resolved Concentrations/Abundances (N_samples x n_components)
        spectra: The resolved Pure Spectra (n_components x M_wavelengths)
            (Note: User snippet return comment said "mcr.ST_opt_ -> The resolved Pure Spectra")
            pymcr convention: D = C @ ST. 
            If D is (N_samples x M_wavelengths), C is (N x K), ST is (K x M).
            So mcr.ST_opt_ is (n_components x M_wavelengths).
    """
    X = _stack_signals(signals)
    
    # MCR requires an initial guess. Using PCA absolute values as suggested.
    pca = PCA(n_components=n_components)
    pca.fit(X)
    initial_spectra = np.abs(pca.components_)

    mcr = McrAR(
        c_regr="nnls",  # Solver for Concentrations (Non-Negative Least Squares)
        st_regr="nnls", # Solver for Spectra (Non-Negative Least Squares)
        c_constraints=[ConstraintNonneg()],
        st_constraints=[ConstraintNonneg(), ConstraintNorm()],
        tol_increase=100.0, # Allow larger error increases to escape local minima
        max_iter=300,
    )
    
    mcr.fit(X, ST=initial_spectra)
    
    if mcr.C_opt_ is None or mcr.ST_opt_ is None:
        raise RuntimeError("MCR-ALS fit failed to produce optimized components")
    c_opt: np.ndarray = mcr.C_opt_  # type: ignore[assignment]
    st_opt: np.ndarray = mcr.ST_opt_  # type: ignore[assignment]
    return c_opt, st_opt
