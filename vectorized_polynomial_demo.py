import numpy as np
import matplotlib.pyplot as plt

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


def compute_piecewise_rate(SINR, M=16):
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




# ============================================================================
# 绘制 M=16 的速率曲线 (SINR 0-100)
# ============================================================================
print("="*70)
print("绘制 16-QAM 速率曲线")
print("="*70)

M = 64
SINR_range = np.linspace(0.1, 1000, 1000)  # 0.1 to 100 线性间隔

# 计算速率

rate= compute_piecewise_rate(SINR_range, M=M)

# 获取阈值
SINR_bar = RATE_PARAMS[M]['SINR_bar']

# 创建图形
plt.figure(figsize=(12, 8))

# 子图1: 实际速率 (bits/symbol)
plt.subplot(2, 1, 1)
plt.plot(SINR_range, rate, 'b-', linewidth=2, label=f'$R_l$ (M={M})')
plt.axvline(x=SINR_bar, color='r', linestyle='--', linewidth=1.5, 
            label=f'Threshold $\\bar{{\\gamma}}$ = {SINR_bar:.2f}')
plt.axhline(y=np.log2(M), color='g', linestyle=':', linewidth=1.5,
            label=f'Max rate = $\\log_2({M})$ = {np.log2(M):.1f} bits/symbol')
plt.grid(True, alpha=0.3)
plt.xlabel('SINR (linear)', fontsize=12)
plt.ylabel('Rate $R_l$ (bits/symbol)', fontsize=12)
plt.legend(fontsize=10)



plt.show()

print(f"\n程序完成！")