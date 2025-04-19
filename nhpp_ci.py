import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'


from scipy.optimize import minimize
from scipy.stats import norm, chi2, poisson

# データ準備
fault_data = np.array([6, 1, 1, 0, 1, 3, 0, 5, 6, 1, 0, 3, 9, 3, 2, 3, 0, 2, 4, 0, 0, 0, 0, 0, 2, 0, 0, 2, 3, 11, 5, 3, 2, 4, 2, 0, 1, 2, 0, 1, 3, 0, 1, 6, 2, 0, 5, 10, 3, 3, 2, 1, 0, 3, 0, 2, 0, 1, 0, 1, 0, 2])
t = np.arange(1, len(fault_data) + 1)
cumulative_faults = np.cumsum(fault_data)

# 対数尤度関数
def log_likelihood(params, t, y):
    a, b = params
    n = len(t)
    logL = 0
    for k in range(1, n):
        delta_y = y[k] - y[k - 1]
        delta_lambda = a * (np.exp(-b * t[k - 1]) - np.exp(-b * t[k]))
        if delta_y > 0 and delta_lambda > 0:
            logL += delta_y * np.log(delta_lambda) - delta_lambda - np.sum(np.log(np.arange(1, delta_y + 1)))
        else:
            logL -= delta_lambda
    return -logL

# 最尤推定
initial_params = [300, 0.009]
bounds = [(1, 1000), (0.001, 0.1)]
res = minimize(log_likelihood, initial_params, args=(t, cumulative_faults), method='L-BFGS-B', bounds=bounds)
mle_a, mle_b = res.x
mle_ll = -res.fun

# Wilks統計量
def wilks_stat(a, t, y, mle_b, mle_ll):
    def neg_log_likelihood_b(b):
        return -log_likelihood([a, b], t, y)
    res_b = minimize(neg_log_likelihood_b, mle_b, bounds=[(0.001, 0.1)], method='L-BFGS-B')
    ll_a = -res_b.fun
    return 2 * (mle_ll - ll_a)

alpha = 0.05
chi2_critical = chi2.ppf(1 - alpha, df=1)
a_values = np.linspace(0.5 * mle_a, 1.5 * mle_a, 100)
wilks_values = np.array([wilks_stat(a, t, cumulative_faults, mle_b, mle_ll) for a in a_values])
ci_wilks = a_values[wilks_values <= chi2_critical]
ci_wilks_lower = np.min(ci_wilks)
ci_wilks_upper = np.max(ci_wilks)

# フィッシャー情報行列
def fisher_information(a, b, t, y):
    n = len(t)
    I = np.zeros((2, 2))
    for k in range(1, n):
        delta_y = y[k] - y[k - 1]
        exp1 = np.exp(-b * t[k - 1])
        exp2 = np.exp(-b * t[k])
        dL_da = exp1 - exp2
        dL_db = a * (-t[k - 1] * exp1 + t[k] * exp2)
        delta_lambda = a * (exp1 - exp2)
        if delta_lambda > 0:
            I[0, 0] += (dL_da ** 2) / delta_lambda
            I[0, 1] += (dL_da * dL_db) / delta_lambda
            I[1, 1] += (dL_db ** 2) / delta_lambda
    I[1, 0] = I[0, 1]
    return I

I = fisher_information(mle_a, mle_b, t, cumulative_faults)
covariance_matrix = np.linalg.inv(I)
se_a = np.sqrt(covariance_matrix[0, 0])
z_critical = norm.ppf(1 - alpha / 2)
ci_fisher_a_lower = mle_a - z_critical * se_a
ci_fisher_a_upper = mle_a + z_critical * se_a

# シミュレーションによる信頼区間
def mean_value_function(t, a, b):
    return a * (1 - np.exp(-b * t))

sample_size = 10000
lower_sim = []
upper_sim = []
mean_sim = []

for t_val in t:
    mean_val = mean_value_function(t_val, mle_a, mle_b)
    samples = poisson.rvs(mean_val, size=sample_size)
    lower_sim.append(np.percentile(samples, 2.5))
    upper_sim.append(np.percentile(samples, 97.5))
    mean_sim.append(mean_val)

# 一般的な正規近似によるCI
Lambda_hat = mean_value_function(t, mle_a, mle_b)
ci_general_lower = Lambda_hat - z_critical * np.sqrt(Lambda_hat)
ci_general_upper = Lambda_hat + z_critical * np.sqrt(Lambda_hat)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(t, cumulative_faults, label='実データ', color='black', linewidth=1)
plt.plot(t, Lambda_hat, label='平均値関数の最尤推定値', color='blue')

# 提案手法
plt.plot(t, lower_sim, linestyle='dotted', color='purple', label='提案手法による信頼区間')
plt.plot(t, upper_sim, linestyle='dotted', color='purple')

# Wilks法
plt.plot(t, ci_wilks_lower * (1 - np.exp(-mle_b * t)), linestyle='dashed', color='green', label='手法2による信頼区間')
plt.plot(t, ci_wilks_upper * (1 - np.exp(-mle_b * t)), linestyle='dashed', color='green')

# Fisher法
plt.plot(t, ci_fisher_a_lower * (1 - np.exp(-mle_b * t)), linestyle='dashdot', color='red', label='手法1による信頼区間')
plt.plot(t, ci_fisher_a_upper * (1 - np.exp(-mle_b * t)), linestyle='dashdot', color='red')

# 一般的な正規近似
plt.plot(t, ci_general_lower, linestyle='dotted', color='orange', label='手法3による信頼区間')
plt.plot(t, ci_general_upper, linestyle='dotted', color='orange')

plt.xlabel('時刻 t')
plt.ylabel('累積故障数')
plt.title('指数型NHPPモデルに基づく信頼区間比較')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
