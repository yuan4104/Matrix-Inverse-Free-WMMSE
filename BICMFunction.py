import numpy as np

# ============================================================================
# Rate Approximation Parameters Table
# ============================================================================
# Parameters for piecewise rate approximation: R_l / log2(M)
# Based on modulation order M and the formula structure
RATE_PARAMS = {
    4: {  # QPSK (4-QAM)
        'SINR_bar': 6.760829753919818,  # threshold SINR value
        'a': 0.028827753381460,
        'b': 1.579459851803511,
        'a_k': [0.028911586230263,-0.091635663780916,-0.059962292900188,0.024971952540583,1.437425381871118,2.081313384915987e-04],  # coefficients for k=0 to 5
    },
    16: {  # 16-QAM
        'SINR_bar': 30.902954325135887,
        'a': 0.314766863216191,
        'b':15.003936906075179,
        'a_k': [-0.003096076898619,0.022856434735149,-0.124149926339006,0.276742999284062,1.157338673346761,1.854053806047828e-04]
    },
    64: {  # 64-QAM
        'SINR_bar': 120.2264434617414,
        'a': 0.887098510455927,
        'b': 45.148109924238870,
        'a_k': [-0.001188038813624,0.005826572542931,-0.016233229294272,0.088510650534479,1.161644818418217,-0.002886977255888]
    },
    256: {  # 256-QAM
        'SINR_bar': 446.6835921509630,
        'a': 3.456128745639584,
        'b': 1.684156449600554e+02,
        'a_k': [-0.001053957564670,0.009856966964429,-0.027994898849416,0.059654398478049,1.186050827586518,-0.012418827670779]
    }
}


def compute_rate(SINR, M=16):
    """
    Compute piecewise rate approximation: R_l / log2(M)
    
    The formula is:
                  ⎧ 1 - a/(SINR - SINR_bar + b)     , SINR > SINR_bar
    R_l/log2(M) = ⎨
                  ⎩ Σ(k=0 to 5) a_k * (ln(1+SINR))^k , SINR ≤ SINR_bar
    

    """
    if M not in RATE_PARAMS:
        raise ValueError(f"Modulation order M={M} not supported. Choose from {list(RATE_PARAMS.keys())}")
    
    params = RATE_PARAMS[M]
    SINR_bar = params['SINR_bar']
    a = params['a']
    b = params['b']
    a_k = np.array(params['a_k'])
    
    # Convert SINR to numpy array for vectorized operations
    SINR = np.asarray(SINR)
    original_shape = SINR.shape
    
    # Initialize output
    rate = np.zeros_like(SINR, dtype=np.float64)
    
    # Case 1: SINR > SINR_bar (high SINR region)
    mask_high = SINR >= SINR_bar
    if np.any(mask_high):
        rate[mask_high] =np.log2(M) *( 1.0 - a / (SINR[mask_high] - SINR_bar + b) )
    
    # Case 2: SINR ≤ SINR_bar (low to medium SINR region)
    mask_low = ~mask_high
    if np.any(mask_low):
        SINR_low = SINR[mask_low]
        x = np.log(1.0 + SINR_low)
        
        # Compute polynomial: sum(p.* [x.^5, x.^4, x.^3, x.^2, x, 1])
        # This matches the fit_QAM function in g.m
        rate_sum = (a_k[0] * x**5 + a_k[1] * x**4 + a_k[2] * x**3 + 
                    a_k[3] * x**2 + a_k[4] * x + a_k[5])
        
        rate[mask_low] =  rate_sum
    
    return rate






# Vectorized version (more efficient)
def compute_interference_covariance_matrix_vectorized(H, P):
    """
    Vectorized computation of interference-plus-noise covariance matrix.
    
    For user i: R_i = I + sum_{k≠i} σ_k^2 H_i P_k (H_i P_k)^H
    
    Parameters:
    -----------
    H : np.ndarray
        Channel matrix with shape (batch, U, Nr, Nt)
    P : np.ndarray
        Precoder matrix with shape (batch, U, Nt, Nr)
    sigma_squared : np.ndarray
        User weights with shape (U,)
    
    Returns:
    --------
    R : np.ndarray
        Interference-plus-noise covariance matrix with shape (batch, U, Nr, Nr)
    """
    batch_size, num_users, nr_ue, nt_bs = H.shape
    
    sigma_squared = np.ones(num_users) 
    
    # Compute H_i @ P_k for all i, k: shape (batch, U, U, Nr, Nr)
    # H[:, i, :, :] @ P[:, k, :, :] for all combinations
    HP = np.einsum('bixy,bkyz->bikxz', H, P)
    # HP[b, i, k, :, :] = H[b, i, :, :] @ P[b, k, :, :]
    
    # Compute (H_i P_k) @ (H_i P_k)^H for all i, k
    HP_HPh = HP @ np.conj(np.transpose(HP, (0, 1, 2, 4, 3)))
    # HP_HPh[b, i, k, :, :] = HP[b, i, k, :, :] @ HP[b, i, k, :, :].conj().T
    
    # Apply user weights σ_k^2
    sigma_weights = sigma_squared.reshape(1, 1, num_users, 1, 1)
    HP_HPh_weighted = HP_HPh * sigma_weights
    
    # Sum over k≠i: create a mask to exclude k=i
    mask = 1 - np.eye(num_users).reshape(1, num_users, num_users, 1, 1)
    interference = np.sum(HP_HPh_weighted * mask, axis=2)
    # interference[b, i, :, :] = sum_{k≠i} σ_k^2 (H_i P_k)(H_i P_k)^H
    
    # Add identity matrix
    I = np.eye(nr_ue).reshape(1, 1, nr_ue, nr_ue)
    R = I + interference
    
    return R


def compute_C_matrix_vectorized(H, P, R):
    """
    Vectorized computation of C matrix for each user.
    
    For user i: C_i = (I + σ_i^{-2} (H_i P_i)^H R_i^{-1} H_i P_i)^{-1}
    
    Parameters:
    -----------
    H : np.ndarray
        Channel matrix with shape (batch, U, Nr, Nt)
    P : np.ndarray
        Precoder matrix with shape (batch, U, Nt, Nr)
    R : np.ndarray
        Interference-plus-noise covariance matrix with shape (batch, U, Nr, Nr)
    sigma_squared : np.ndarray
        User weights with shape (U,)
    
    Returns:
    --------
    C : np.ndarray
        C matrix with shape (batch, U, Nr, Nr)
        C[b, i, :, :] is the C matrix for user i in batch b
    """
    batch_size, num_users, nr_ue, nt_bs = H.shape
    sigma_squared = np.ones(num_users) 
    
    
    # Compute H_i @ P_i for each user i (diagonal terms only)
    # HP_i[b, i, :, :] = H[b, i, :, :] @ P[b, i, :, :]
    HP_i = np.einsum('bixy,biyz->bixz', H, P)
    # HP_i shape: (batch, U, Nr, Nr)
    
    # Compute (H_i P_i)^H = conjugate transpose
    HP_i_H = np.conj(np.transpose(HP_i, (0, 1, 3, 2)))
    # HP_i_H shape: (batch, U, Nr, Nr)
    
    # Compute R_i^{-1} for each user i
    R_inv = np.linalg.inv(R)
    # R_inv shape: (batch, U, Nr, Nr)
    
    # Compute (H_i P_i)^H @ R_i^{-1} @ H_i P_i for each user i
    # First: R_i^{-1} @ H_i P_i
    R_inv_HP = R_inv @ HP_i
    # R_inv_HP shape: (batch, U, Nr, Nr)
    
    # Second: (H_i P_i)^H @ (R_i^{-1} @ H_i P_i)
    HP_H_R_inv_HP = HP_i_H @ R_inv_HP
    # HP_H_R_inv_HP shape: (batch, U, Nr, Nr)
    
    # Apply σ_i^{-2} scaling
    sigma_inv_squared = 1.0 / sigma_squared  # shape: (U,)
    sigma_weights = sigma_inv_squared.reshape(1, num_users, 1, 1)
    scaled_term = sigma_weights * HP_H_R_inv_HP
    # scaled_term shape: (batch, U, Nr, Nr)
    
    # Add identity matrix: I + σ_i^{-2} (H_i P_i)^H R_i^{-1} H_i P_i
    I = np.eye(nr_ue).reshape(1, 1, nr_ue, nr_ue)
    term_to_invert = I + scaled_term
    # term_to_invert shape: (batch, U, Nr, Nr)
    
    # Compute C_i = (I + ...)^{-1}
    C = np.linalg.inv(term_to_invert)
    # C shape: (batch, U, Nr, Nr)
    
    return C


def SINR(C):
    """
    Compute [C_i]_{i,i}^{-1} - 1 for each diagonal element of C_i.
    
    For each user i and each antenna j: result[b, i, j] = C[b, i, j, j]^{-1} - 1
    
    This is equivalent to computing the inverse of each diagonal element,
    then subtracting 1 from it.
    
    Parameters:
    -----------
    C : np.ndarray
        C matrix with shape (batch, U, Nr, Nr)
    
    Returns:
    --------
    result : np.ndarray
        Diagonal inverse minus identity with shape (batch, U, Nr)
        result[b, i, j] = 1/C[b, i, j, j] - 1
    """
    batch_size, num_users, nr_ue, _ = C.shape
    
    # Extract diagonal elements of C
    # C_diag[b, i, j] = C[b, i, j, j]
    C_diag = np.diagonal(C, axis1=2, axis2=3)
    # C_diag shape: (batch, U, Nr)
    
    # Compute inverse of diagonal elements: 1 / C[b, i, j, j]
    C_diag_inv = 1.0 / C_diag
    # C_diag_inv shape: (batch, U, Nr)
    
    # Subtract 1: C_diag^{-1} - 1
    result = C_diag_inv - 1.0
    # result shape: (batch, U, Nr)
    
    return np.real(result)


def BICM_rate(H,P, M=16):
    """
    """
    if M not in RATE_PARAMS:
        raise ValueError(f"Modulation order M={M} not supported. Choose from {list(RATE_PARAMS.keys())}")
    R = compute_interference_covariance_matrix_vectorized(H, P)
    C = compute_C_matrix_vectorized(H, P, R)
    sinr = SINR(C)
    rate = compute_rate(sinr, M=M)
    return np.mean(rate)



if __name__ == "__main__":
    U = 4  # number of users
    Nt = 8  # number of BS antennas
    Nr = 2  # number of UE antennas
    batch = 12  # batch size

    # Generate random channel and precoder matrices
    H = np.random.randn(batch, U, Nr, Nt) + 1j * np.random.randn(batch, U, Nr, Nt)
    P = np.random.randn(batch, U, Nt, Nr) + 1j * np.random.randn(batch, U, Nt, Nr)


    rate_16QAM = BICM_rate(H,P, M=4)
    print("Rate shape:", np.mean(rate_16QAM))  # Average rate over batch and antennas
    


