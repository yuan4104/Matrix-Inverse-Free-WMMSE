"""
Complex-Real Block Matrix Transformation for 4D Arrays
Author: Generated for matrix transformation utilities
Date: 2026.1.7

For a 4D complex matrix X with shape (d1, d2, m, n), transform the last two dimensions
into a real-valued block matrix Y with shape (d1, d2, 2m, 2n) using the mapping:
    Y = [real(X), -imag(X)]
        [imag(X),  real(X)]

And provide the inverse transformation from Y back to X.
"""

import numpy as np


def complex_to_real_block(X):
    """
    Convert a 4D complex matrix to real block matrix form.
    
    For a complex matrix X with shape (d1, d2, m, n), creates a real matrix Y
    with shape (d1, d2, 2m, 2n) where:
        Y = [real(X), -imag(X)]
            [imag(X),  real(X)]
    
    Parameters:
    -----------
    X : np.ndarray
        Complex array with shape (d1, d2, m, n)
    
    Returns:
    --------
    Y : np.ndarray
        Real array with shape (d1, d2, 2m, 2n)
    
    Example:
    --------
    >>> X = np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)
    >>> Y = complex_to_real_block(X)
    >>> Y.shape
    (2, 3, 8, 10)
    >>> np.allclose(X, real_block_to_complex(Y))
    True
    """
    if not np.iscomplexobj(X):
        raise ValueError("Input X must be a complex array")
    
    d1, d2, m, n = X.shape
    
    # Extract real and imaginary parts
    X_real = np.real(X)  # shape: (d1, d2, m, n)
    X_imag = np.imag(X)  # shape: (d1, d2, m, n)
    
    # Build the block matrix: first row [real, -imag], second row [imag, real]
    top_row = np.concatenate([X_real, -X_imag], axis=3)     # shape: (d1, d2, m, 2n)
    bottom_row = np.concatenate([X_imag, X_real], axis=3)   # shape: (d1, d2, m, 2n)
    
    Y = np.concatenate([top_row, bottom_row], axis=2)       # shape: (d1, d2, 2m, 2n)
    
    return Y


def real_block_to_complex(Y):
    """
    Convert a real block matrix back to complex matrix form (inverse transformation).
    
    For a real matrix Y with shape (d1, d2, 2m, 2n) structured as:
        Y = [Y11, Y12]
            [Y21, Y22]
    where each block has shape (m, n), reconstructs the complex matrix X with
    shape (d1, d2, m, n) such that:
        X = Y11 + 1j * Y21
    
    This assumes Y was created using complex_to_real_block, i.e.:
        Y11 = real(X), Y12 = -imag(X), Y21 = imag(X), Y22 = real(X)
    
    Parameters:
    -----------
    Y : np.ndarray
        Real array with shape (d1, d2, 2m, 2n)
    
    Returns:
    --------
    X : np.ndarray
        Complex array with shape (d1, d2, m, n)
    
    Example:
    --------
    >>> X_orig = np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)
    >>> Y = complex_to_real_block(X_orig)
    >>> X_reconstructed = real_block_to_complex(Y)
    >>> np.allclose(X_orig, X_reconstructed)
    True
    """
    if np.iscomplexobj(Y):
        raise ValueError("Input Y must be a real array")
    
    d1, d2, two_m, two_n = Y.shape
    
    if two_m % 2 != 0 or two_n % 2 != 0:
        raise ValueError("Last two dimensions of Y must be even (got shape {})".format(Y.shape))
    
    m = two_m // 2
    n = two_n // 2
    
    # Extract the four blocks
    Y11 = Y[:, :, :m, :n]       # real(X)
    Y12 = Y[:, :, :m, n:]       # -imag(X)
    Y21 = Y[:, :, m:, :n]       # imag(X)
    Y22 = Y[:, :, m:, n:]       # real(X) (redundant, should equal Y11)
    
    # Reconstruct complex matrix: X = real(X) + 1j * imag(X)
    X = Y11 + 1j * Y21
    
    return X


# Test and demonstration
if __name__ == '__main__':
    print("="*60)
    print("Testing Complex ↔ Real Block Matrix Transformations")
    print("="*60)
    
    # Create a random 4D complex matrix
    d1, d2, m, n = 2, 3, 4, 5
    X_original = np.random.randn(d1, d2, m, n) + 1j * np.random.randn(d1, d2, m, n)
    
    print(f"\n1. Original complex matrix X shape: {X_original.shape}")
    print(f"   X[0,0,:2,:2] =\n{X_original[0, 0, :2, :2]}")
    
    # Forward transformation: complex → real block
    Y = complex_to_real_block(X_original)
    print(f"\n2. Transformed real block matrix Y shape: {Y.shape}")
    print(f"   Y is real: {not np.iscomplexobj(Y)}")
    print(f"   Y[0,0,:4,:4] (top-left 4×4 block) =\n{Y[0, 0, :4, :4]}")
    
    # Inverse transformation: real block → complex
    X_reconstructed = real_block_to_complex(Y)
    print(f"\n3. Reconstructed complex matrix X shape: {X_reconstructed.shape}")
    print(f"   X_reconstructed[0,0,:2,:2] =\n{X_reconstructed[0, 0, :2, :2]}")
    
    # Verify reconstruction accuracy
    reconstruction_error = np.max(np.abs(X_original - X_reconstructed))
    print(f"\n4. Reconstruction verification:")
    print(f"   Max absolute error: {reconstruction_error:.2e}")
    print(f"   Reconstruction successful: {np.allclose(X_original, X_reconstructed)}")
    
    # Demonstrate the block structure for a simple case
    print(f"\n5. Block structure demonstration:")
    print(f"   For a single complex number z = a + bi, the block is:")
    print(f"   [ a  -b]")
    print(f"   [ b   a]")
    
    z = 3 + 4j
    X_simple = np.array([[[[z]]]]) # shape (1,1,1,1)
    Y_simple = complex_to_real_block(X_simple)
    print(f"\n   z = {z}")
    print(f"   Block representation:")
    print(f"{Y_simple[0, 0, :, :]}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
