"""Heart AI Lab — 연구용 Cleveland 이진 분류 실험실.

근거: Akella & Akella (2021), DOI 10.2144/fsoa-2020-0206;
UCI Heart Disease, DOI 10.24432/C52P4X;
scikit-learn threshold tuning / common pitfalls / Pipeline documentation.
논문 아이디어를 확장한 독립 Python 구현이며, 논문의 정확한 재현이 아니다.
실행: python -m streamlit run main.py (Python 3.12 권장)
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_VERSION = "1.1.0"
SEED = 2026  # 비교 중 seed shopping 방지: UI에서 변경하지 않는다.
DATA_URL = "https://archive.ics.uci.edu/static/public/45/data.csv"
DATA_SHA = "868cd646a53ffbc55611192bae7496911a163937c185710f6b8aa11549b88df1"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
LABELS = dict(zip(FEATURES, ["나이 (년)", "성별 (원자료 코드)", "흉통 유형",
    "안정 시 혈압 (mmHg)", "혈청 콜레스테롤 (mg/dL)", "공복혈당 >120 mg/dL 여부",
    "안정 시 심전도", "운동 중 최대 심박수", "운동 유발 협심증", "운동 시 ST 하강",
    "운동 시 ST 기울기", "형광투시로 관찰된 주요 혈관 수", "탈륨 검사 결과"]))
MODELS = {"LR": "로지스틱 회귀", "DT": "의사결정나무", "RF": "랜덤포레스트"}
BASE = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]
PRESETS = {"전체 13개": FEATURES, "기본 정보·검사 7개": BASE,
           "운동검사 포함 11개": [f for f in FEATURES if f not in ("ca", "thal")]}
RATE_KEYS = ["전체 오류율", "정확도", "민감도", "특이도", "위음성률", "위양성률", "양성예측도"]
SOURCES = [
    ("UCI Heart Disease", "https://doi.org/10.24432/C52P4X",
     "공식 Cleveland 데이터·변수 사전·num 이진화. 데이터 CC BY 4.0."),
    ("Detrano et al. (1989)", "https://pubmed.ncbi.nlm.nih.gov/2756873/",
     "관상동맥조영술 대상 환자 자료의 배경과 집단 간 일반화 한계."),
    ("Akella & Akella (2021)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8147740/",
     "Cleveland 모델 비교, LR 기준 임계값 0.5, RF 기본 나무 수 500을 참고."),
    ("scikit-learn: threshold tuning", "https://scikit-learn.org/stable/modules/classification_threshold.html",
     "학습 점수와 최종 판정 분리. 임계값 탐색은 개발 자료에서만 수행."),
    ("scikit-learn: common pitfalls", "https://scikit-learn.org/stable/common_pitfalls.html",
     "Pipeline을 이용한 학습 구간 내부 전처리와 평가 자료 분리."),
    ("scikit-learn: cross-validation", "https://scikit-learn.org/stable/modules/cross_validation.html",
     "층화 5-fold와 학습에서 제외된 사례의 OOF 예측."),
]


def parse_data(raw: bytes) -> pd.DataFrame:
    """실험용 원자료를 고정한다. 다른 데이터/잘못된 인코딩을 조용히 허용하지 않는다."""
    if hashlib.sha256(raw).hexdigest() != DATA_SHA:
        raise ValueError("공식 데이터의 저장된 SHA-256과 다릅니다. 원본 파일·출처를 확인하세요.")
    df = pd.read_csv(io.BytesIO(raw), na_values=["?", "", "NA"])
    if df.shape != (303, 14) or df.columns.tolist() != FEATURES + ["num"]:
        raise ValueError("Cleveland 303×14 스키마가 아닙니다.")
    df = df.apply(pd.to_numeric, errors="raise")
    if df["num"].isna().any() or not df["num"].isin(range(5)).all():
        raise ValueError("목표 변수 num이 잘못되었습니다.")
    return df


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    local = Path(__file__).parent / "data" / "cleveland_uci.csv"
    if local.exists():
        return parse_data(local.read_bytes())
    # main.py와 requirements.txt 두 파일만 업로드해도 공식 서버에서 읽는다.
    req = Request(DATA_URL, headers={"User-Agent": "HeartAILab/1.0 research"})
    with urlopen(req, timeout=25) as response:
        raw = response.read()
    return parse_data(raw)


def split_ids(df: pd.DataFrame):
    y = (df["num"] > 0).astype(int)
    return train_test_split(np.arange(len(df)), test_size=0.20,
                            stratify=y, random_state=SEED)


def make_pipeline(model: str, features: list[str], params: dict) -> Pipeline:
    if not features or not set(features).issubset(FEATURES):
        raise ValueError("유효한 입력 변수를 하나 이상 선택하세요.")
    numeric = [f for f in features if f in NUMERIC]
    categorical = [f for f in features if f not in NUMERIC]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    if model == "LR":
        estimator = LogisticRegression(C=params["C"], max_iter=5000,
                                       solver="lbfgs", random_state=SEED)
    elif model == "DT":
        estimator = DecisionTreeClassifier(max_depth=params["depth"],
            min_samples_leaf=params["leaf"], random_state=SEED)
    elif model == "RF":
        estimator = RandomForestClassifier(n_estimators=params["trees"],
            max_depth=params["depth"], min_samples_leaf=params["leaf"],
            random_state=SEED, n_jobs=1)
    else:
        raise ValueError("지원하지 않는 모델입니다.")
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", estimator)])


@st.cache_data(show_spinner=False, max_entries=12)
def run_experiment(df: pd.DataFrame, config_json: str) -> dict:
    """개발 자료만으로 OOF 예측. 테스트 라벨/예측은 반환하지 않는다."""
    cfg = json.loads(config_json)
    dev_ids, test_ids = split_ids(df)
    X = df.loc[dev_ids, cfg["features"]].reset_index(drop=True)
    y = (df.loc[dev_ids, "num"] > 0).astype(int).to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    predictions, fold_numbers = {}, np.zeros(len(y), dtype=int)
    for number, (_, val) in enumerate(folds, 1):
        fold_numbers[val] = number
    for name in cfg["models"]:
        pipeline = make_pipeline(name, cfg["features"], cfg["params"])
        oof = np.full(len(y), np.nan)
        for train, val in folds:
            fitted = clone(pipeline).fit(X.iloc[train], y[train])
            oof[val] = fitted.predict_proba(X.iloc[val])[:, 1]
        if not np.isfinite(oof).all():
            raise ValueError("평가 예측값에 결측/무한대가 있습니다.")
        predictions[name] = oof
    return {"y": y, "scores": predictions, "dev_ids": dev_ids,
            "folds": fold_numbers, "test_n": len(test_ids), "config": cfg}


def metrics(y, scores, threshold: float) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a / b) if b else float("nan")
    return {"전체 오류율": ratio(fp + fn, len(y)), "정확도": ratio(tp + tn, len(y)),
            "민감도": ratio(tp, tp + fn), "특이도": ratio(tn, tn + fp),
            "위음성률": ratio(fn, tp + fn), "위양성률": ratio(fp, tn + fp),
            "양성예측도": ratio(tp, tp + fp),
            "ROC-AUC": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan"),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "N": len(y)}


def threshold_table(y, scores, thresholds) -> pd.DataFrame:
    """모든 임계값의 혼동행렬을 한 번에 계산; 임계값과 무관한 AUC는 한 번만 계산."""
    y, scores, thresholds = np.asarray(y), np.asarray(scores), np.asarray(thresholds)
    predicted = scores[:, None] >= thresholds[None, :]
    positive = y[:, None] == 1
    tp = (predicted & positive).sum(axis=0)
    fp = (predicted & ~positive).sum(axis=0)
    fn = (positive & ~predicted).sum(axis=0)
    tn = (~positive & ~predicted).sum(axis=0)
    def divide(a, b):
        return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=np.asarray(b) != 0)
    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")
    return pd.DataFrame({"임계값": thresholds, "전체 오류율": divide(fp+fn, len(y)),
        "정확도": divide(tp+tn, len(y)), "민감도": divide(tp, tp+fn),
        "특이도": divide(tn, tn+fp), "위음성률": divide(fn, tp+fn),
        "위양성률": divide(fp, tn+fp), "양성예측도": divide(tp, tp+fp),
        "ROC-AUC": auc, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "N": len(y)})


def select_threshold(y, scores, target: float) -> tuple[float, dict]:
    """개발 OOF에서 목표 민감도를 만족하는 가장 큰 임계값 → 최소 위양성률.
    점수가 같을 때 >= 규칙을 일관되게 사용한다. 테스트에서 목표 달성 보장 아님.
    """
    candidates = np.unique(np.r_[0., scores, 1.])
    table = threshold_table(y, scores, candidates)
    feasible = table.loc[table["민감도"] >= target]
    chosen = float(feasible.iloc[-1]["임계값"])
    return chosen, metrics(y, scores, chosen)


def final_evaluate(df: pd.DataFrame, cfg: dict, model: str, threshold: float) -> dict:
    dev_ids, test_ids = split_ids(df)
    estimator = make_pipeline(model, cfg["features"], cfg["params"])
    estimator.fit(df.loc[dev_ids, cfg["features"]], (df.loc[dev_ids, "num"] > 0).astype(int))
    scores = estimator.predict_proba(df.loc[test_ids, cfg["features"]])[:, 1]
    y = (df.loc[test_ids, "num"] > 0).astype(int).to_numpy()
    return {"model": model, "threshold": threshold, "config": cfg,
            "metrics": metrics(y, scores, threshold), "y": y, "scores": scores,
            "ids": test_ids, "time": datetime.now(timezone.utc).isoformat()}


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p, denom = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0., center - half), min(1., center + half)


def rate_table(rows):
    table = pd.DataFrame(rows)
    formats = {k: "{:.1%}" for k in RATE_KEYS if k in table}
    if "ROC-AUC" in table:
        formats["ROC-AUC"] = "{:.3f}"
    return table.style.format(formats, na_rep="정의 불가")


def show_matrix(m: dict, key: str):
    fig = px.imshow([[m["TN"], m["FP"]], [m["FN"], m["TP"]]],
        x=["예측 음성", "예측 양성"], y=["실제 음성", "실제 양성"],
        text_auto=True, color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(height=320, margin=dict(t=15, b=20), coloraxis_showscale=False,
                      dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig, use_container_width=True, key=key, config=PLOT_CONFIG)
    st.caption(f"위음성: {m['FN']}/{m['TP']+m['FN']}명 · 위양성: {m['FP']}/{m['TN']+m['FP']}명")


def source_panel():
    st.subheader("논문·공식 문서와 구현의 연결")
    for title, url, applied in SOURCES:
        st.markdown(f"**[{title}]({url})** — {applied}")
    st.markdown("""
**논문과 달라진 설계**

- 원논문은 결측 사례 6개를 제외했다. 이 앱은 303개를 유지하고 각 fold의 학습 자료에서만 결측값을 대체한다.
- 원논문의 R 코드·분할·6종 모델을 그대로 복제하지 않았다. Python의 3종 모델, 고정 80:20 분할,
  개발 자료 5-fold OOF 평가를 사용한다. 논문의 수치를 이 앱의 실행 결과처럼 표시하지 않는다.
- 명목형 코드를 연속 수치로 간주하지 않고 one-hot 처리한다. 숫자형은 학습 fold에서 표준화한다.
- 판정 규칙은 `score >= threshold`이다. 논문 LR 설명의 `> 0.5`와 경계에서 다를 수 있다.
- 모델의 출력값은 보정되지 않은 예측 점수다. 실제 질환 확률로 단정하지 않는다.

**해석의 범위**

과거 검사 대상자의 관상동맥질환 분류 연구용이다. 일반인의 미래 심장질환 위험이나 개인 진단에
적용하지 않는다. `ca`, `thal` 등 전문 검사 정보가 포함되어 있어 문진만으로 조기진단했다는
결론을 내릴 수 없다. 변수 제거 효과는 인과관계가 아니다. 작은 표본, 단일 출처, 여러 설정의
반복 탐색 때문에 탐색 최고 점수는 낙관적일 수 있다. 독립 외부 검증은 수행하지 않았다.

**데이터 저작자·라이선스**

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. Heart Disease [Dataset].
UCI Machine Learning Repository. DOI: 10.24432/C52P4X.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
원본 CSV는 변경 없이 사용하고 앱 내부에서 목표 이진화·학습용 전처리를 수행한다.
""")


# ---- 화면 개선 버전 1.1.0: 아래는 기존 과학·ML 계산 로직을 재사용합니다. ----
PLOT_CONFIG = {
    "displayModeBar": False, "scrollZoom": False,
    "doubleClick": False, "displaylogo": False, "responsive": True,
}


def fix_fraction_axes(fig, *, x_numeric=True):
    """터치·드래그·더블탭 확대를 제한하고 그래프 범위를 고정합니다."""
    fig.update_layout(dragmode=False, transition_duration=0,
                      margin=dict(l=8, r=8, t=30, b=16))
    if x_numeric:
        fig.update_xaxes(range=[0, 1], autorange=False, fixedrange=True,
                         tick0=0, dtick=0.2)
    else:
        fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(range=[0, 1], autorange=False, fixedrange=True,
                     tickformat=".0%", tick0=0, dtick=0.2)
    return fig


def degenerate_prediction_warning(m: dict) -> str | None:
    """민감도 100%가 환자 구분 성공을 의미하지 않는 상황을 알립니다."""
    if m["TP"] + m["FP"] == m["N"]:
        return ("모든 사례를 양성으로 판정했습니다. 민감도는 100%이지만, "
                "음성 사례도 모두 양성이므로 특이도는 0%입니다. 실질적인 구분 성능으로 해석하지 마세요.")
    if m["TN"] + m["FN"] == m["N"]:
        return ("모든 사례를 음성으로 판정했습니다. 위양성은 없지만 "
                "실제 양성을 모두 놓쳤습니다. 실질적인 구분 성능으로 해석하지 마세요.")
    return None


def format_case_delta(n: int, what: str) -> str:
    if n > 0:
        return f"{what} {n}명 증가"
    if n < 0:
        return f"{what} {-n}명 감소"
    return f"{what} 변화 없음"


def compare_to_baseline(current: dict, baseline: dict, current_threshold: float) -> str:
    fn = current["FN"] - baseline["FN"]
    fp = current["FP"] - baseline["FP"]
    if abs(current_threshold - 0.5) < 1e-9:
        return "기준 임계값 0.50입니다. 다른 값으로 조절하면 두 오류의 증감이 표시됩니다."
    direction = "낮추었습니다" if current_threshold < 0.5 else "높였습니다"
    return (f"임계값을 0.50에서 {current_threshold:.2f}로 {direction}. "
            f"{format_case_delta(fn, '위음성')}; {format_case_delta(fp, '위양성')}. "
            "동일한 개발 데이터에서 판정 규칙만 바꾼 결과이며 모델은 재학습되지 않았습니다.")


def display_history(history: list[dict]) -> pd.DataFrame:
    """읽기 쉬운 순서로 요약; 전체 세부 설정은 CSV에 보존합니다."""
    view = pd.DataFrame(history)
    if view.empty:
        return view
    preferred = ["기록ID", "실험명", "모델", "변수수", "임계값", "전체 오류율",
                 "위음성률", "위양성률", "FN", "FP", "민감도", "특이도", "ROC-AUC",
                 "기록시각_UTC", "가설_관찰"]
    return view[[c for c in preferred if c in view.columns]].copy()


def history_comparison(a: dict, b: dict):
    """두 기록의 차이를 계산하고 변경된 입력 변수 목록을 반환합니다."""
    important = ["전체 오류율", "민감도", "특이도", "위음성률", "위양성률"]
    rows = [{"지표": metric, "실험 A": float(a[metric]),
             "실험 B": float(b[metric]), "변화(B-A)": float(b[metric]) - float(a[metric])}
            for metric in important]
    for metric in ["FN", "FP"]:
        rows.append({"지표": metric, "실험 A": int(a[metric]),
                     "실험 B": int(b[metric]), "변화(B-A)": int(b[metric]) - int(a[metric])})
    f_a, f_b = set(str(a["입력변수"]).split(",")), set(str(b["입력변수"]).split(","))
    return pd.DataFrame(rows), sorted(f_b - f_a), sorted(f_a - f_b)


def main():
    st.set_page_config(page_title="Heart AI Lab", page_icon="🫀", layout="wide",
                       initial_sidebar_state="expanded")
    st.title("🫀 Heart AI Lab")
    st.markdown("**같은 데이터, 다른 판단.** 모델·입력 변수·임계값에 따른 오류율 변화를 비교합니다.")
    st.caption("의학교육·통계 탐구용 / 개인 의료 진단 불가 / UCI Cleveland / 검증 근거는 ‘자료·방법’ 탭")

    try:
        df = load_data()
    except Exception as exc:
        st.error(f"원본 데이터를 읽지 못했습니다: {exc}")
        st.info("ZIP에 포함된 data/cleveland_uci.csv 파일을 GitHub에도 올려 주세요. 가상 자료로 대체하지 않습니다.")
        return

    for key, value in [("history", []), ("run", None), ("final", None),
                       ("chart_revision", 0)]:
        if key not in st.session_state:
            st.session_state[key] = value

    st.sidebar.header("실험 설정")
    preset = st.sidebar.selectbox("입력 변수군", list(PRESETS), key="preset")
    if st.session_state.get("last_preset") != preset:
        st.session_state["features"] = PRESETS[preset].copy()
        st.session_state["last_preset"] = preset
    st.sidebar.caption("변수군을 선택하거나 아래에서 개별 변수를 변경하세요.")

    with st.sidebar.form("training"):
        names = st.multiselect("비교할 머신러닝 모델", list(MODELS),
                               default=list(MODELS), format_func=MODELS.get, key="models")
        submitted = st.form_submit_button("학습 및 비교", type="primary",
                                          use_container_width=True)
        with st.expander("개별 입력 변수 선택 (13개)", expanded=False):
            features = st.multiselect("사용할 학습 변수", FEATURES,
                                      format_func=lambda f: f"{f} · {LABELS[f]}",
                                      key="features")
            st.caption("변수를 바꾼 후 위의 '학습 및 비교'를 누르면 재학습됩니다.")
        with st.expander("고급 모델 설정", expanded=False):
            C = st.select_slider("로지스틱 회귀의 C", [0.01, 0.1, 1., 10., 100.],
                                 value=1., key="C")
            trees = st.select_slider("랜덤포레스트 나무 수", [100, 300, 500],
                                     value=500, key="trees")
            depth = st.selectbox("나무 최대 깊이", [3, 5, 8, None], index=1,
                                 format_func=lambda x: "제한 없음" if x is None else str(x),
                                 key="depth")
            leaf = st.slider("나무 말단 최소 사례 수", 1, 10, 2, key="leaf")
    st.sidebar.caption("임계값만 바꾸면 재학습하지 않습니다. 모델·변수·설정을 바꾸면 학습 버튼을 누르세요.")
    cfg_new = {"features": [f for f in FEATURES if f in features],
               "models": [m for m in MODELS if m in names],
               "params": {"C": C, "trees": trees, "depth": depth, "leaf": leaf}}

    if submitted:
        if not features or not names:
            st.sidebar.error("머신러닝 모델과 입력 변수를 각각 하나 이상 선택하세요.")
        else:
            try:
                with st.spinner("개발 자료 5-fold 교차검증으로 학습하고 있습니다…"):
                    st.session_state["run"] = run_experiment(df, json.dumps(cfg_new, sort_keys=True))
            except Exception as exc:
                st.error(f"모델 학습 실패: {exc}. 이전 결과는 유지합니다.")

    tabs = st.tabs(["임계값 실험", "모델·기록 비교", "최종 평가", "자료·방법"])
    with tabs[3]:
        source_panel()
        st.subheader("13개 입력 변수와 결측값 처리")
        st.dataframe(pd.DataFrame({"변수": FEATURES, "설명": [LABELS[f] for f in FEATURES],
            "처리 방식": ["중앙값 대체·표준화" if f in NUMERIC else "최빈값 대체·원핫 인코딩" for f in FEATURES],
            "결측값 수": [int(df[f].isna().sum()) for f in FEATURES]}),
            hide_index=True, use_container_width=True)
        st.code(f"데이터 SHA-256: {DATA_SHA}\n분할 시드: {SEED}\n버전: {APP_VERSION}\nscikit-learn: {sklearn.__version__}")

    run = st.session_state["run"]
    if run is None:
        with tabs[0]:
            st.info("왼쪽에서 조건을 선택한 뒤 ‘학습 및 비교’를 눌러 첫 실험을 시작하세요.")
            st.markdown("**추천 탐구 순서:** 임계값 0.30·0.50·0.70 비교 → 모델 간 비교 → 변수군 변경 → 기록 간 비교 → 최종 평가")
        return

    cfg = run["config"]  # 현재 편집 중인 선택지가 아니라 실제 학습 완료 조건을 사용한다.
    y = run["y"]
    run_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    if st.session_state["final"] is not None:
        st.warning("최종 테스트 결과는 이미 공개되었습니다. 이후의 모델 탐색·임계값 변경은 새로운 독립 검증이 아닙니다.")

    # 탭 밖에서 정의: 어떤 탭을 열어도 현재 비교용 임계값과 목표가 유지됩니다.
    if st.session_state.get("focus") not in cfg["models"]:
        st.session_state["focus"] = cfg["models"][0]

    with tabs[0]:
        st.caption(f"실험 {run_id} · 개발 OOF {len(y)}명 · 최종 보관 {run['test_n']}명 · 학습 변수 {len(cfg['features'])}개")
        with st.expander("학습에 실제로 사용한 변수 확인", expanded=False):
            st.write(" · ".join(f"{f} ({LABELS[f]})" for f in cfg["features"]))
        focus = st.selectbox("자세히 분석할 모델", cfg["models"],
                             format_func=MODELS.get, key="focus")
        threshold = st.slider("양성 판정 임계값 (예측 점수 ≥ 임계값)",
                              0., 1., 0.50, 0.01, key="threshold")
        score = run["scores"][focus]
        current = metrics(y, score, threshold)
        baseline = metrics(y, score, 0.50)
        for col, name in zip(st.columns(5), ["전체 오류율", "민감도", "특이도", "위음성률", "위양성률"]):
            col.metric(name, f"{current[name]:.1%}")
        st.info(compare_to_baseline(current, baseline, threshold))
        a, b = st.columns(2)
        with a:
            st.metric("위음성 (놓친 양성)", f"{current['FN']}명",
                      delta=f"{current['FN']-baseline['FN']:+d}명 (0.50 대비)",
                      delta_color="inverse")
        with b:
            st.metric("위양성 (잘못 분류한 음성)", f"{current['FP']}명",
                      delta=f"{current['FP']-baseline['FP']:+d}명 (0.50 대비)",
                      delta_color="inverse")
        warning = degenerate_prediction_warning(current)
        if warning:
            st.warning(warning)

        left, right = st.columns([1, 1.35])
        with left:
            st.markdown("**현재 판정의 혼동행렬**")
            show_matrix(current, f"oof_matrix_{st.session_state['chart_revision']}")
        with right:
            st.markdown("**임계값별 오류율 변화**")
            curve = threshold_table(y, score, np.linspace(0, 1, 101))
            fig = px.line(curve, x="임계값", y=["전체 오류율", "위음성률", "위양성률"],
                          labels={"value": "오류 비율", "variable": "지표"})
            fig.add_vline(x=threshold, line_dash="dash", line_color="black")
            fix_fraction_axes(fig)
            fig.update_layout(legend_title_text="", legend=dict(orientation="h", y=1.15, x=0))
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG,
                            key=f"threshold_curve_{st.session_state['chart_revision']}")
        if st.button("그래프 초기화", key="reset_chart"):
            st.session_state["chart_revision"] += 1
            st.rerun()
        st.caption("화면 확대·드래그는 제한되어 있습니다. 축 범위는 임계값 0~1, 오류율 0~100%로 고정됩니다.")

        st.subheader("대표 임계값 0.30·0.50·0.70 비교")
        representative = threshold_table(y, score, [0.30, 0.50, 0.70])
        st.dataframe(rate_table(representative[["임계값", "전체 오류율", "민감도", "특이도",
                                            "위음성률", "위양성률", "FN", "FP"]]),
                     use_container_width=True, hide_index=True)
        st.caption("위음성률=FN/(FN+TP), 위양성률=FP/(FP+TN). 임계값만 바꿀 때 ROC-AUC는 변하지 않습니다.")

        hypothesis = st.text_input("실험 가설·관찰 메모", key="hypothesis",
                                   placeholder="예: 임계값을 낮추면 위음성은 줄지만 위양성은 늘어나는가?")
        experiment_name = st.text_input("실험 이름 (선택)", key="experiment_name",
                                        placeholder="예: 전체 13개 변수·기본 설정")
        with st.expander("5개 평가 구간의 변동 확인", expanded=False):
            folds = []
            for k in range(1, 6):
                mask = run["folds"] == k
                folds.append({"fold": k, **metrics(y[mask], score[mask], threshold)})
            st.dataframe(rate_table(folds), use_container_width=True, hide_index=True)
            st.caption("전체 OOF 지표는 5개 구간의 예측을 합쳐 계산하며, 개별 fold는 서로 독립된 반복 연구가 아닙니다.")

        if st.button("현재 조건의 모든 모델 결과 기록", key="save"):
            label = experiment_name.strip() or f"실험 {len(st.session_state['history'])//max(1,len(run['scores']))+1}"
            for model, scores in run["scores"].items():
                record = {"기록ID": len(st.session_state["history"]) + 1,
                          "실험명": label, "기록시각_UTC": datetime.now(timezone.utc).isoformat(),
                          "실행": run_id, "평가": "개발 OOF", "모델코드": model,
                          "모델": MODELS[model], "변수수": len(cfg["features"]),
                          "임계값": threshold, "가설_관찰": hypothesis,
                          "입력변수": ",".join(cfg["features"]),
                          "설정": json.dumps(cfg["params"], ensure_ascii=False),
                          "최종공개이후": st.session_state["final"] is not None,
                          "seed": SEED, "data_sha256": DATA_SHA,
                          "sklearn": sklearn.__version__,
                          **metrics(y, scores, threshold)}
                st.session_state["history"].append(record)
            st.success(f"{len(run['scores'])}개 모델의 결과를 저장했습니다. ‘모델·기록 비교’ 탭에서 비교할 수 있습니다.")

    with tabs[1]:
        st.subheader(f"같은 임계값 {threshold:.2f}에서 모델 비교")
        rows = [{"모델": MODELS[name], **metrics(y, scores, threshold)}
                for name, scores in run["scores"].items()]
        st.dataframe(rate_table(rows), hide_index=True, use_container_width=True)
        fig = px.bar(pd.DataFrame(rows), x="모델", y=["전체 오류율", "위음성률", "위양성률"],
                     barmode="group", labels={"value": "비율", "variable": "지표"})
        fix_fraction_axes(fig, x_numeric=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key="models_bar")
        roc = go.Figure()
        for model, scores in run["scores"].items():
            fpr, tpr, _ = roc_curve(y, scores)
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{MODELS[model]} (AUC {roc_auc_score(y, scores):.3f})"))
            mm = metrics(y, scores, threshold)
            roc.add_trace(go.Scatter(x=[mm["위양성률"]], y=[mm["민감도"]],
                                     mode="markers", marker=dict(size=11),
                                     name=f"{MODELS[model]} 현재 임계값"))
        roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="gray"))
        roc.update_layout(xaxis_title="위양성률", yaxis_title="민감도")
        fix_fraction_axes(roc)
        st.plotly_chart(roc, use_container_width=True, config=PLOT_CONFIG, key="roc")
        st.caption("ROC 곡선 위의 점은 현재 임계값의 위치입니다. 임계값을 움직여도 같은 모델의 AUC는 변하지 않습니다.")

        st.subheader("목표 민감도 이상을 만족하는 조건 비교")
        target = st.slider("탐구용 목표 민감도 (임상 기준 아님)",
                           0.70, 1.00, 0.90, 0.01, key="target")
        suggested, target_rows, warnings = {}, [], []
        for model, scores in run["scores"].items():
            t, mm = select_threshold(y, scores, target)
            suggested[model] = t
            warning = degenerate_prediction_warning(mm)
            if warning:
                warnings.append(MODELS[model])
            target_rows.append({"모델": MODELS[model], "선택 임계값": t,
                                "판정 상태": "전원 양성" if mm["TP"]+mm["FP"] == mm["N"] else
                                "전원 음성" if mm["TN"]+mm["FN"] == mm["N"] else "일반 판정",
                                **mm})
        st.dataframe(rate_table(target_rows), hide_index=True, use_container_width=True)
        if warnings:
            st.warning(" / ".join(warnings) + ": 현재 목표를 충족시키는 선택값에서 전체를 한쪽으로 판정합니다. "
                       "특히 전원 양성은 민감도 100%지만 특이도 0%로, 유용한 분류가 아닙니다.")
        st.caption("개발 OOF에서 목표 이상을 만족시키는 가장 높은 임계값을 선택합니다. "
                   "실제 달성 민감도는 모델마다 다를 수 있으며, 최종 테스트 성능은 보장되지 않습니다.")

        st.subheader("실험 기록")
        history = st.session_state["history"]
        if history:
            full = pd.DataFrame(history)
            st.dataframe(rate_table(display_history(history)), hide_index=True,
                         use_container_width=True)
            st.download_button("전체 실험 기록 CSV", full.to_csv(index=False).encode("utf-8-sig"),
                               file_name="heart_ai_experiments.csv", mime="text/csv", key="download_history")
            if len(history) >= 2:
                st.markdown("**두 실험 직접 비교**")
                ids = list(range(len(history)))
                def fmt_row(i):
                    rec = history[i]
                    return (f"#{rec.get('기록ID', i+1)} · {rec.get('실험명','기록')} · "
                            f"{rec.get('모델','모델')} · 변수 {rec.get('변수수','?')}개 · "
                            f"임계값 {float(rec['임계값']):.2f}")
                c1, c2 = st.columns(2)
                with c1:
                    a_idx = st.selectbox("실험 A (기준)", ids,
                                         index=max(0, len(ids)-2), format_func=fmt_row,
                                         key="history_A")
                with c2:
                    b_idx = st.selectbox("실험 B (비교)", ids, index=len(ids)-1,
                                         format_func=fmt_row, key="history_B")
                a, b = history[a_idx], history[b_idx]
                if a_idx == b_idx:
                    st.info("서로 다른 기록을 선택하면 차이가 표시됩니다.")
                else:
                    change, added, removed = history_comparison(a, b)
                    change_readable = change.copy().astype({"실험 A": object, "실험 B": object, "변화(B-A)": object})
                    for i, item in change.iterrows():
                        if item["지표"] in RATE_KEYS:
                            change_readable.loc[i, "실험 A"] = f"{item['실험 A']:.1%}"
                            change_readable.loc[i, "실험 B"] = f"{item['실험 B']:.1%}"
                            change_readable.loc[i, "변화(B-A)"] = f"{100*item['변화(B-A)']:+.1f}%p"
                        else:
                            change_readable.loc[i, "실험 A"] = f"{int(item['실험 A'])}명"
                            change_readable.loc[i, "실험 B"] = f"{int(item['실험 B'])}명"
                            change_readable.loc[i, "변화(B-A)"] = f"{int(item['변화(B-A)']):+d}명"
                    st.dataframe(change_readable, use_container_width=True, hide_index=True)
                    st.write("**실험 B에서 추가된 변수:** " + (", ".join(added) or "없음"))
                    st.write("**실험 B에서 제외된 변수:** " + (", ".join(removed) or "없음"))
                    if (a.get("모델") != b.get("모델") or
                            a.get("설정") != b.get("설정") or
                            a.get("임계값") != b.get("임계값") or
                            a.get("실행") != b.get("실행")):
                        st.caption("두 기록은 모델·설정·임계값·학습 변수 등이 동시에 달라질 수 있습니다. "
                                   "차이를 특정 한 변수의 인과효과로 해석하지 마세요.")
        else:
            st.info("먼저 ‘임계값 실험’ 탭에서 결과를 기록해 주세요.")
        export = pd.DataFrame({"원자료행번호_0부터": run["dev_ids"],
                               "fold": run["folds"], "실제양성": y})
        for model, scores in run["scores"].items():
            export[f"{model}_score"] = scores
        st.download_button("현재 개발 OOF 예측 CSV", export.to_csv(index=False).encode("utf-8-sig"),
                           file_name="heart_ai_oof.csv", mime="text/csv", key="download_oof")

    with tabs[2]:
        st.subheader("최종 설정을 확정한 후 한 번 평가")
        st.write("여기서는 개발에 사용하지 않은 20% 데이터를 최초 확정된 조건으로 한 번만 평가합니다.")
        st.caption("다른 세션에서 반복하면 같은 데이터에 다시 맞출 위험이 있습니다. 테스트 결과를 보고 모델·임계값을 재선정하지 마세요.")
        if st.session_state["final"] is None:
            chosen = st.selectbox("최종 모델", cfg["models"], format_func=MODELS.get,
                                  key="final_model")
            mode = st.radio("최종 임계값 선택",
                            ["현재 슬라이더 값", "목표 민감도 기준 OOF 선택값"], key="final_mode")
            final_t = threshold if mode == "현재 슬라이더 값" else suggested[chosen]
            st.write(f"**확정할 조건:** {MODELS[chosen]} · 임계값 {final_t:.6f} · 변수 {len(cfg['features'])}개")
            consent = st.checkbox("설정을 확정했고, 테스트 결과를 본 뒤 다시 최적화하면 독립 검증이 아님을 이해했습니다.",
                                  key="confirm")
            if st.button("확정하고 최종 테스트 공개", disabled=not consent, key="finalize"):
                try:
                    with st.spinner("개발 데이터 전체 재학습 후 보관된 테스트 데이터를 평가합니다…"):
                        st.session_state["final"] = final_evaluate(df, cfg, chosen, float(final_t))
                    st.rerun()
                except Exception as exc:
                    st.error(f"최종 평가 실패: {exc}")
        if st.session_state["final"] is not None:
            result = st.session_state["final"]
            fm = result["metrics"]
            st.success(f"최초 확정 결과: {MODELS[result['model']]} / 임계값 {result['threshold']:.6f}")
            st.write("학습 변수: " + ", ".join(result["config"]["features"]))
            st.dataframe(rate_table([fm]), hide_index=True, use_container_width=True)
            warning = degenerate_prediction_warning(fm)
            if warning:
                st.warning(warning)
            show_matrix(fm, "final_matrix")
            intervals = []
            for label, k, n in [("정확도", fm["TP"]+fm["TN"], fm["N"]),
                                ("민감도", fm["TP"], fm["TP"]+fm["FN"]),
                                ("특이도", fm["TN"], fm["TN"]+fm["FP"])]:
                low, high = wilson(k, n)
                intervals.append({"지표": label, "분자/분모": f"{k}/{n}",
                                  "95% Wilson 구간": f"{low:.1%}~{high:.1%}"})
            st.dataframe(intervals, hide_index=True, use_container_width=True)
            st.caption("이 구간은 평가 표본 내 비율의 불확실성만 나타냅니다. 데이터 편향·모델 선택·외부 일반화 문제는 포함하지 않습니다.")
            report = {"app_version": APP_VERSION, "data_url": DATA_URL,
                      "data_sha256": DATA_SHA, "seed": SEED,
                      "python": platform.python_version(), "sklearn": sklearn.__version__,
                      **{k: v for k, v in result.items() if k not in ("y", "scores", "ids")},
                      "test_rows": result["ids"].tolist(),
                      "test_labels": result["y"].tolist(),
                      "test_scores": result["scores"].tolist(), "sources": SOURCES}
            report["metrics"] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                 for k, v in fm.items()}
            st.download_button("최종 결과·조건 JSON",
                               json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                               file_name="heart_ai_final.json", mime="application/json")


if __name__ == "__main__":
    main()
