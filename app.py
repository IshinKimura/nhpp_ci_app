import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, chi2, poisson

# --- 日本語フォント（必要なら） ---
plt.rcParams['font.family'] = 'MS Gothic'

# --- サンプルデータ ---
fault_data = np.array([6, 1, 1, 0, 1, 3, 0, 5, 6, 1, 0, 3, 9, 3, 2, 3, 0, 2, 4, 0, 0, 0, 0, 0, 2, 0, 0, 2, 3, 11, 5, 3, 2, 4, 2, 0, 1, 2, 0, 1, 3, 0, 1, 6, 2, 0, 5, 10, 3, 3, 2, 1, 0, 3, 0, 2, 0, 1, 0, 1, 0, 2])
t = np.arange(1, len(fault_data) + 1)
y = np.cumsum(fault_data)

# --- 関数定義 ---
def mean_value_function(t, a, b):
    return a * (1 - np.exp(-b * t))

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

def wilks_stat(a, t, y, mle_b, mle_ll):
    def neg_log_likelihood_b(b):
        return -log_likelihood([a, b], t, y)
    res_b = minimize(neg_log_likelihood_b, mle_b, bounds=[(0.001, 0.1)])
    ll_a = -res_b.fun
    return 2 * (mle_ll - ll_a)

# --- Streamlit アプリ ---
st.markdown("## 📊 指数型NHPPモデル：信頼区間の可視化ツール")


methods = st.multiselect(
    "信頼区間の構築手法を選んでください",
    ["Wilksの尤度比検定", "フィッシャー情報行列（近似正規）", "シミュレーション", "一般的な正規近似"],
    default=["フィッシャー情報行列（近似正規）"]
)

if st.button("信頼区間を計算して表示"):
    # --- MLE推定 ---
    init = [300, 0.009]
    bounds = [(1, 1000), (0.001, 0.1)]
    res = minimize(log_likelihood, init, args=(t, y), method='L-BFGS-B', bounds=bounds)
    a_hat, b_hat = res.x
    mle_ll = -res.fun
    Lambda_hat = mean_value_function(t, a_hat, b_hat)

    # --- グラフ作成 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, y, label="実データ", color='black')
    ax.plot(t, Lambda_hat, label="平均値関数", color='blue')

    # --- Wilks ---
    if "Wilksの尤度比検定" in methods:
        a_vals = np.linspace(0.5 * a_hat, 1.5 * a_hat, 100)
        chi2_crit = chi2.ppf(0.95, df=1)
        wilks_vals = np.array([wilks_stat(a, t, y, b_hat, mle_ll) for a in a_vals])
        ci_a = a_vals[wilks_vals <= chi2_crit]
        if len(ci_a) > 0:
            a_lower, a_upper = np.min(ci_a), np.max(ci_a)
            ax.plot(t, mean_value_function(t, a_lower, b_hat), color='green', linestyle='--', label="Wilks下限")
            ax.plot(t, mean_value_function(t, a_upper, b_hat), color='green', linestyle='--', label="Wilks上限")

    # --- Fisher ---
    if "フィッシャー情報行列（近似正規）" in methods:
        I = fisher_information(a_hat, b_hat, t, y)
        cov = np.linalg.inv(I)
        se_a = np.sqrt(cov[0, 0])
        z = norm.ppf(0.975)
        a_lower = a_hat - z * se_a
        a_upper = a_hat + z * se_a
        ax.plot(t, mean_value_function(t, a_lower, b_hat), color='red', linestyle='-.', label="近似正規下限")
        ax.plot(t, mean_value_function(t, a_upper, b_hat), color='red', linestyle='-.', label="近似正規上限")

    # --- Simulation ---
    if "シミュレーション" in methods:
        sim_lower = []
        sim_upper = []
        for t_val in t:
            mu = mean_value_function(t_val, a_hat, b_hat)
            samples = poisson.rvs(mu, size=10000)
            sim_lower.append(np.percentile(samples, 2.5))
            sim_upper.append(np.percentile(samples, 97.5))
        ax.plot(t, sim_lower, color='purple', linestyle=':', label="シミュ下限")
        ax.plot(t, sim_upper, color='purple', linestyle=':', label="シミュ上限")

    # --- General Normal Approximation ---
    if "一般的な正規近似" in methods:
        z = norm.ppf(0.975)
        ci_lower = Lambda_hat - z * np.sqrt(Lambda_hat)
        ci_upper = Lambda_hat + z * np.sqrt(Lambda_hat)
        ax.plot(t, ci_lower, color='orange', linestyle=':', label="正規近似下限")
        ax.plot(t, ci_upper, color='orange', linestyle=':', label="正規近似上限")

    ax.set_xlabel("時刻 t")
    ax.set_ylabel("累積故障数")
    ax.set_title("信頼区間の比較（選択手法）")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
