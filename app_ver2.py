import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, chi2, poisson
import matplotlib.font_manager as fm
import io

# ──────────────────────────────
# 日本語フォント設定（フォールバック方式）
# ──────────────────────────────
def _set_fallback_japanese_font() -> None:
    _JA_FONTS = [
        "IPAexGothic",        # Cloud で fonts-ipaexfont を入れると使える
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in _JA_FONTS:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    else:
        plt.rcParams["font.family"] = "sans-serif"
        st.warning(
            "⚠️ 日本語フォントが見つかりません。文字化けする場合は "
            "`packages.txt` に `fonts-ipaexfont` を追加し、再デプロイしてください。"
        )
    plt.rcParams["axes.unicode_minus"] = False  # − が化けないように

try:
    # japanize-matplotlib が入っていれば自動設定される
    import japanize_matplotlib  # noqa: F401
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    # 無ければフォールバック
    _set_fallback_japanese_font()

# ──────────────────────────────
# アプリタイトル
# ──────────────────────────────
st.markdown(
    """
    <h2 style='text-align: center; white-space: nowrap; font-size: 24px;'>
        📊 指数型NHPPモデル：信頼区間の可視化ツール
    </h2>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────
# デフォルト故障データ
# ──────────────────────────────
DEFAULT_FAULTS = (
    "6, 1, 1, 0, 1, 3, 0, 5, 6, 1, 0, 3, 9, 3, 2, 3, 0, 2, 4, 0, 0, 0, 0, 0, "
    "2, 0, 0, 2, 3, 11, 5, 3, 2, 4, 2, 0, 1, 2, 0, 1, 3, 0, 1, 6, 2, 0, 5, 10, "
    "3, 3, 2, 1, 0, 3, 0, 2, 0, 1, 0, 1, 0, 2"
)

# ──────────────────────────────
# データ入力
# ──────────────────────────────
data_input_method = st.radio(
    "故障データの入力方法を選んでください", ["手入力", "CSVアップロード"]
)

if data_input_method == "手入力":
    data_text = st.text_area(
        "非累積の故障データをカンマ区切りで入力してください（例：6,1,1,0,...）",
        value=DEFAULT_FAULTS,
        height=120,
    )
    try:
        fault_data = np.array(
            [int(x.strip()) for x in data_text.split(",") if x.strip() != ""]
        )
    except Exception:
        st.error("入力に誤りがあります。半角のカンマ区切りで整数を入力してください。")
        st.stop()

elif data_input_method == "CSVアップロード":
    uploaded_file = st.file_uploader(
        "CSVファイルをアップロード（1列目のみ使用）", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            fault_data = np.loadtxt(io.StringIO(content), delimiter=",", dtype=int)
        except Exception:
            st.error(
                "CSVファイルの読み込みに失敗しました。1列目に数値が並んでいるCSVにしてください。"
            )
            st.stop()
    else:
        st.stop()

# ──────────────────────────────
# UI：信頼度・手法選択
# ──────────────────────────────
confidence_level = st.slider("信頼区間の信頼度（%）", 30, 99, 95)
alpha = 1 - confidence_level / 100

methods = st.multiselect(
    "信頼区間の構築手法を選んでください",
    [
        "Wilksの尤度比検定統計量",
        "近似正規検定統計量",
        "シミュレーション",
        "一般的な正規近似",
    ],
    default=["一般的な正規近似"],
)

# ──────────────────────────────
# モデル関連関数
# ──────────────────────────────
def mean_value_function(t, a, b):
    return a * (1 - np.exp(-b * t))


def log_likelihood(params, t, y):
    a, b = params
    logL = 0
    for k in range(1, len(t)):
        delta_y = y[k] - y[k - 1]
        delta_lambda = a * (np.exp(-b * t[k - 1]) - np.exp(-b * t[k]))
        if delta_y > 0 and delta_lambda > 0:
            logL += (
                delta_y * np.log(delta_lambda)
                - delta_lambda
                - np.sum(np.log(np.arange(1, delta_y + 1)))
            )
        else:
            logL -= delta_lambda
    return -logL


def fisher_information(a, b, t, y):
    I = np.zeros((2, 2))
    for k in range(1, len(t)):
        exp1 = np.exp(-b * t[k - 1])
        exp2 = np.exp(-b * t[k])
        dL_da = exp1 - exp2
        dL_db = a * (-t[k - 1] * exp1 + t[k] * exp2)
        delta_lambda = a * (exp1 - exp2)
        if delta_lambda > 0:
            I[0, 0] += (dL_da**2) / delta_lambda
            I[0, 1] += (dL_da * dL_db) / delta_lambda
            I[1, 1] += (dL_db**2) / delta_lambda
    I[1, 0] = I[0, 1]
    return I


def wilks_stat(a, t, y, mle_b, mle_ll):
    def neg_log_likelihood_b(b):
        return -log_likelihood([a, b], t, y)

    res_b = minimize(neg_log_likelihood_b, mle_b, bounds=[(0.001, 0.1)])
    ll_a = -res_b.fun
    return 2 * (mle_ll - ll_a)


# ──────────────────────────────
# 計算 & 描画
# ──────────────────────────────
if st.button("信頼区間を計算して表示"):
    t = np.arange(1, len(fault_data) + 1)
    y = np.cumsum(fault_data)

    # --- MLE 推定
    res = minimize(
        log_likelihood,
        [max(y), 0.01],
        args=(t, y),
        bounds=[(1, 10000), (0.0001, 0.1)],
    )
    a_hat, b_hat = res.x
    mle_ll = -res.fun
    Lambda_hat = mean_value_function(t, a_hat, b_hat)

    # --- グラフ準備
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, y, label="実データ", color="black")
    ax.plot(t, Lambda_hat, label="平均値関数", color="blue")

    # --- Wilks
    if "Wilksの尤度比検定統計量" in methods:
        a_vals = np.linspace(0.5 * a_hat, 1.5 * a_hat, 100)
        chi2_crit = chi2.ppf(1 - alpha, df=1)
        wilks_vals = np.array(
            [wilks_stat(a, t, y, b_hat, mle_ll) for a in a_vals]
        )
        ci_a = a_vals[wilks_vals <= chi2_crit]
        if ci_a.size > 0:
            a_lower, a_upper = ci_a.min(), ci_a.max()
            ax.plot(
                t,
                mean_value_function(t, a_lower, b_hat),
                linestyle="--",
                color="green",
                label="Wilks下限",
            )
            ax.plot(
                t,
                mean_value_function(t, a_upper, b_hat),
                linestyle="--",
                color="green",
                label="Wilks上限",
            )

    # --- Fisher
    if "近似正規検定統計量" in methods:
        I = fisher_information(a_hat, b_hat, t, y)
        cov = np.linalg.inv(I)
        se_a = np.sqrt(cov[0, 0])
        z = norm.ppf(1 - alpha / 2)
        a_lower = a_hat - z * se_a
        a_upper = a_hat + z * se_a
        ax.plot(
            t,
            mean_value_function(t, a_lower, b_hat),
            linestyle="-.",
            color="red",
            label="Fisher下限",
        )
        ax.plot(
            t,
            mean_value_function(t, a_upper, b_hat),
            linestyle="-.",
            color="red",
            label="Fisher上限",
        )

    # --- シミュレーション
    if "シミュレーション" in methods:
        lower_sim, upper_sim = [], []
        for t_val in t:
            mu = mean_value_function(t_val, a_hat, b_hat)
            samples = poisson.rvs(mu, size=5000)  # 回数を抑えて高速化
            lower_sim.append(np.percentile(samples, 100 * alpha / 2))
            upper_sim.append(np.percentile(samples, 100 * (1 - alpha / 2)))
        ax.plot(t, lower_sim, linestyle=":", color="purple", label="シミュ下限")
        ax.plot(t, upper_sim, linestyle=":", color="purple", label="シミュ上限")

    # --- 正規近似
    if "一般的な正規近似" in methods:
        z = norm.ppf(1 - alpha / 2)
        ci_lower = Lambda_hat - z * np.sqrt(Lambda_hat)
        ci_upper = Lambda_hat + z * np.sqrt(Lambda_hat)
        ax.plot(t, ci_lower, linestyle=":", color="orange", label="正規近似下限")
        ax.plot(t, ci_upper, linestyle=":", color="orange", label="正規近似上限")

    # --- 仕上げ
    ax.set_xlabel("時刻 t")
    ax.set_ylabel("累積故障数")
    ax.set_title(f"信頼区間の比較（信頼度 {confidence_level}%）")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
