"""
DSBench-mini — a faithful, offline-gradeable slice of DSBench's *data analysis*
track (Jing et al., ICLR 2025).

Faithful to the paper's methodology: each task is a long natural-language intro
+ a self-contained tabular dataset + a multiple-choice or fill-in question with a
single discrete ground-truth answer. The agent must write & run code (python_exec)
to compute the answer — guessing from the intro alone fails. Grading is
deterministic exact/numeric match (DSBench uses an LLM judge only to normalize
phrasing of discrete answers; we check them directly, which the paper notes is
equivalent for discrete GTs).

Reference points (DSBench Table 4, data-analysis accuracy): Llama3-70B ≈ 23.4%,
GPT-4o ≈ 28%, AutoGen+GPT-4o ≈ 34%, human ≈ 64%.

Run:  python benchmarks/dsbench_mini.py
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from _bench import Task, extract_final_text, make_client, run_suite

DS_SYSTEM = (
    "You are a data analyst. Use the python_exec tool to compute the answer — "
    "read any data files with pandas from the current directory. Do NOT guess; "
    "compute from the data. When done, give your final answer on the last line "
    "in exactly this form: 'Answer: <value>' (a single choice letter for "
    "multiple-choice, or a single number for fill-in)."
)


# ── answer extraction / grading helpers ──────────────────────────────────────
def _answer_field(text: str) -> str | None:
    m = list(re.finditer(r"answer\s*[:=]\s*(.+)", text, flags=re.IGNORECASE))
    return m[-1].group(1).strip() if m else None


def grade_mc(choices: list, correct_value):
    """Grade a multiple-choice answer by VALUE (faithful to DSBench's semantic
    LLM judge for discrete answers): accept either the correct choice letter OR
    the correct choice value, however the agent phrases it."""
    letters = "ABCDE"
    correct_letter = letters[choices.index(correct_value)]
    numeric = all(isinstance(c, (int, float)) for c in choices)

    def _g(result, workdir):
        text = extract_final_text(result)
        field = _answer_field(text) or text
        # explicit choice letter at the start of the answer field
        ml = re.match(r"\s*\(?([A-Ea-e])\b", field)
        got_letter = ml.group(1).upper() if ml else None
        if numeric:
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", field.replace(",", ""))]
            if nums and min(choices, key=lambda c: abs(c - nums[-1])) == correct_value:
                return 1.0, f"value={nums[-1]}→{correct_value} (={correct_letter})"
        else:  # string-bucket choices
            if str(correct_value).lower() in field.lower():
                return 1.0, f"value≈{correct_value!r} (={correct_letter})"
        if got_letter == correct_letter:
            return 1.0, f"letter={got_letter}"
        return 0.0, f"got={field[:28]!r} want={correct_value}({correct_letter})"
    return _g


def grade_numeric(gt: float, tol: float):
    def _g(result, workdir):
        text = extract_final_text(result)
        field = _answer_field(text) or text
        nums = re.findall(r"-?\d+(?:\.\d+)?", field.replace(",", ""))
        if not nums:
            return 0.0, f"no number found; want≈{gt}"
        got = float(nums[-1])
        ok = abs(got - gt) <= tol
        return (1.0 if ok else 0.0, f"got={got} want={gt}±{tol}")
    return _g


def _mc_block(choices: list, correct_value):
    """Render 'A. v0\\nB. v1 ...' and return (block, grader)."""
    letters = "ABCDE"
    block = "\n".join(f"{letters[i]}. {v}" for i, v in enumerate(choices))
    return block, grade_mc(choices, correct_value)


def _write_csv(workdir, name, df):
    df.to_csv(os.path.join(workdir, name), index=False)


# ════════════════════════════════════════════════════════════════════════════
# Tasks
# ════════════════════════════════════════════════════════════════════════════

def _t_voters() -> Task:
    rng = np.random.default_rng(11)
    codes = rng.integers(105, 195, size=1000)
    df = pd.DataFrame({"voter_id": np.arange(1, 1001),
                       "age": rng.integers(18, 90, size=1000),
                       "district_code": codes})
    gt = int(((codes >= 135) & (codes <= 144)).sum())
    choices = [gt - 2, gt - 1, gt, gt + 1, gt + 2]
    block, grader = _mc_block(choices, gt)
    intro = ("The country of Excelstan has 1000 registered voters. Each voter has a "
             "District Code from 105 to 194 that maps to one of nine districts in "
             "10-code bands: Alpha=105-114, Beta=115-124, Gamma=125-134, "
             "Delta=135-144, Epsilon=145-154, Zeta=155-164, Eta=165-174, "
             "Theta=175-184, Iota=185-194. The file ./voters.csv has columns "
             "voter_id, age, district_code.")
    return Task(
        id="voters_district",
        prompt=f"<introduction>{intro}</introduction>\n"
               f"<question>How many voters are in the Delta District?</question>\n{block}",
        grade=grader,
        setup=lambda wd: _write_csv(wd, "voters.csv", df),
        meta={"type": "mc", "gt": gt})


def _t_inflation() -> Task:
    gt = round(70 * 1.02 ** 8, 2)  # 82.02
    choices = [81.94, 81.96, 81.98, 82.00, gt]
    block, grader = _mc_block(choices, gt)
    intro = ("An electricity supply contract sets a price 'cap' that starts at "
             "$70.00 per MWh in calendar year 2017 and increases by exactly 2% per "
             "year, compounding annually, on each 1 January thereafter.")
    return Task(
        id="inflation_cap",
        prompt=f"<introduction>{intro}</introduction>\n"
               f"<question>What is the cap price in April 2025? (USD per MWh)</question>\n{block}",
        grade=grader, meta={"type": "mc", "gt": gt})


def _t_coin_mc() -> Task:
    # P(>=7 heads in 10 fair flips) = 176/1024 = 0.1719 -> bucket "15-20%"
    choices = ["<10%", "10-15%", "15-20%", ">20%"]
    block, grader = _mc_block(choices, "15-20%")
    intro = ("A fair coin (P(heads)=0.5) is flipped 10 times, and we record the "
             "number of heads. Estimate, e.g. by Monte Carlo simulation with many "
             "trials, the probability of getting AT LEAST 7 heads.")
    return Task(
        id="coin_montecarlo",
        prompt=f"<introduction>{intro}</introduction>\n"
               f"<question>What is the probability of at least 7 heads?</question>\n{block}",
        grade=grader, meta={"type": "mc"})


def _t_feb_average() -> Task:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
    base = 0.5 + 0.4 * np.sin(np.arange(len(ts)) / 24) ** 2
    kwh = np.round(base + rng.normal(0, 0.05, len(ts)), 4)
    df = pd.DataFrame({"timestamp": ts.astype(str), "kwh": kwh}).sample(frac=1, random_state=3)
    feb = kwh[(ts.month == 2)]
    gt = round(float(feb.mean()), 3)
    choices = [round(gt - 0.024, 3), round(gt - 0.012, 3), gt, round(gt + 0.012, 3)]
    block, grader = _mc_block(choices, gt)
    intro = ("The file ./meter.csv contains hourly smart-meter electricity readings "
             "for all of 2023, with columns timestamp (an ISO datetime string) and "
             "kwh. The rows are NOT in chronological order.")
    return Task(
        id="feb_avg_usage",
        prompt=f"<introduction>{intro}</introduction>\n<question>What is the average "
               f"kwh per hour during February 2023?</question>\n{block}",
        grade=grader,
        setup=lambda wd: _write_csv(wd, "meter.csv", df), meta={"type": "mc", "gt": gt})


def _t_diabetes_coef() -> Task:
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    b = load_diabetes(as_frame=True)
    df = b.frame
    model = LinearRegression().fit(df.drop(columns=["target"]), df["target"])
    coef = dict(zip(df.drop(columns=["target"]).columns, model.coef_))
    gt = round(float(coef["bmi"]))
    intro = ("The file ./diabetes.csv has 10 standardized physiological features "
             "(age, sex, bmi, bp, s1..s6) and a numeric 'target' (disease progression). ")
    return Task(
        id="diabetes_ols_coef",
        prompt=f"<introduction>{intro}</introduction>\n<question>Fit an ordinary "
               f"least-squares linear regression of target on all 10 features. What is "
               f"the fitted coefficient on the 'bmi' feature, rounded to the nearest "
               f"integer?</question>\nGive a single number.",
        grade=grade_numeric(gt, tol=8),
        setup=lambda wd: _write_csv(wd, "diabetes.csv", df), meta={"type": "num", "gt": gt})


def _t_titanic() -> Task:
    # deterministic synthetic passengers with controlled survival by group
    rows = []
    rng = np.random.default_rng(5)
    spec = {(1, "female"): (100, 90), (1, "male"): (120, 40),
            (2, "female"): (90, 70), (2, "male"): (110, 20),
            (3, "female"): (150, 60), (3, "male"): (200, 25)}
    for (pclass, sex), (n, surv) in spec.items():
        s = np.array([1] * surv + [0] * (n - surv))
        rng.shuffle(s)
        for v in s:
            rows.append({"pclass": pclass, "sex": sex,
                         "age": int(rng.integers(1, 80)), "survived": int(v)})
    df = pd.DataFrame(rows).sample(frac=1, random_state=2).reset_index(drop=True)
    gt = round(100 * 90 / 100)  # female 1st class survival rate = 90%
    choices = [85, 88, 90, 92, 95]
    block, grader = _mc_block(choices, gt)
    intro = ("The file ./passengers.csv lists ship passengers with columns pclass "
             "(1/2/3), sex ('male'/'female'), age, and survived (1=survived, 0=died).")
    return Task(
        id="titanic_survival",
        prompt=f"<introduction>{intro}</introduction>\n<question>What is the survival "
               f"rate (%) of female passengers in 1st class (pclass=1), to the nearest "
               f"percent?</question>\n{block}",
        grade=grader,
        setup=lambda wd: _write_csv(wd, "passengers.csv", df), meta={"type": "mc", "gt": gt})


def _t_iris_ratio() -> Task:
    from sklearn.datasets import load_iris
    b = load_iris(as_frame=True)
    df = b.frame.copy()
    df["species"] = b.target_names[b.target]
    df["_ratio"] = df["petal length (cm)"] / df["petal width (cm)"]
    gt = df.groupby("species")["_ratio"].mean().idxmax()  # 'setosa'
    df = df.drop(columns=["_ratio"])
    species = list(b.target_names)
    block, grader = _mc_block(species, gt)
    intro = ("The file ./iris.csv has iris flower measurements: 'sepal length (cm)', "
             "'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', and "
             "'species' (setosa/versicolor/virginica).")
    return Task(
        id="iris_petal_ratio",
        prompt=f"<introduction>{intro}</introduction>\n<question>For which species is "
               f"the mean ratio of petal length to petal width the largest?</question>\n{block}",
        grade=grader,
        setup=lambda wd: _write_csv(wd, "iris.csv", df), meta={"type": "mc", "gt": gt})


def _t_correlation() -> Task:
    rng = np.random.default_rng(13)
    temp = rng.uniform(0, 40, 500)
    count = 3.0 * temp + rng.normal(0, 15, 500)
    df = pd.DataFrame({"temp": np.round(temp, 2), "count": np.round(count, 1)})
    gt = round(float(np.corrcoef(temp, count)[0, 1]), 2)
    intro = ("The file ./rides.csv has 500 rows with columns temp (temperature, °C) "
             "and count (number of bike rentals that hour).")
    return Task(
        id="pearson_correlation",
        prompt=f"<introduction>{intro}</introduction>\n<question>What is the Pearson "
               f"correlation coefficient between temp and count, rounded to 2 "
               f"decimals?</question>\nGive a single number.",
        grade=grade_numeric(gt, tol=0.03),
        setup=lambda wd: _write_csv(wd, "rides.csv", df), meta={"type": "num", "gt": gt})


def _t_join() -> Task:
    rng = np.random.default_rng(21)
    customers = pd.DataFrame({"customer_id": np.arange(1, 51),
                              "region": rng.choice(["West", "East", "South"], 50)})
    orders = pd.DataFrame({"order_id": np.arange(1, 301),
                           "customer_id": rng.integers(1, 51, 300),
                           "amount": np.round(rng.uniform(10, 500, 300), 2)})
    merged = orders.merge(customers, on="customer_id")
    gt = round(float(merged.loc[merged.region == "West", "amount"].sum()))
    choices = sorted({gt, gt + 137, gt - 212, gt + 401, gt - 88})
    # ensure exactly the correct value is present and 5 options
    choices = [gt - 212, gt - 88, gt, gt + 137, gt + 401]
    block, grader = _mc_block(choices, gt)
    intro = ("Two files: ./orders.csv (order_id, customer_id, amount) and "
             "./customers.csv (customer_id, region). Each order belongs to a customer; "
             "each customer is in one region.")
    return Task(
        id="multitable_revenue",
        prompt=f"<introduction>{intro}</introduction>\n<question>What is the total order "
               f"amount (sum of 'amount') from customers in the West region, rounded to "
               f"the nearest dollar?</question>\n{block}",
        grade=grader,
        setup=lambda wd: (_write_csv(wd, "orders.csv", orders),
                          _write_csv(wd, "customers.csv", customers)),
        meta={"type": "mc", "gt": gt})


def build_tasks() -> list[Task]:
    return [_t_voters(), _t_inflation(), _t_coin_mc(), _t_feb_average(),
            _t_diabetes_coef(), _t_titanic(), _t_iris_ratio(), _t_correlation(), _t_join()]


def make_agent(task, workdir, react: bool = False):
    from genai_studio.agents import Agent
    from genai_studio.agents.datascience.tools import make_python_exec
    from genai_studio.agents.trace import JsonlTracer
    return Agent(client=make_client(react=react), tools=[make_python_exec()],
                 system=DS_SYSTEM,
                 tracer=JsonlTracer(os.path.join(workdir, "trace.jsonl")),
                 max_steps=8)


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tasks = build_tasks()[:limit]
    run_suite("DSBench-mini", tasks, make_agent)
