# NEW SHAP BRANCH

import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

# Constants
DECISION_THRESHOLD = 0.1275  # Youden-tuned threshold for class 1 (mortality)

# Try to import the SHAP + traditional score helper that returns a Matplotlib figure
try:
    from predict_and_explain import predict_proba_and_shap
    HAS_HELPER = True
except Exception:
    HAS_HELPER = False

SCHEMA_PATH = "feature_schema.json"
PIPELINE_PATH = "saluSCORE_ped_pipeline.pkl"


@st.cache_resource
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def reset_calculator_state():
    """Clear patient input + UI flags but keep page routing and disclaimer acceptance."""
    keep = {"page", "accepted_disclaimer"}
    for k in list(st.session_state.keys()):
        if k in keep:
            continue
        # form field keys and UI flags used in this app
        if (
            k.startswith("fld_")           # text/radio/select inputs
            or k.startswith("surg_")       # surgery multiselects per group
            or k.startswith("btn_")        # analyze / impute buttons
            or k in {"show_missing_prompt", "_scroll_to_form_top_once"}
        ):
            del st.session_state[k]


def disclaimer_gate(disclaimer_brief: str,
                    disclaimer_full: str,
                    owner="SaluSCORE™ Project Team",
                    version="v0.1",
                    log_to_csv=False) -> bool:
    """
    Show brief terms first; on 'Read Full Terms of Use' swap to full text.
    Block app until 'I Accept and Proceed' is pressed, then hide permanently.
    """
    st.markdown("""
    <style>
    /* Make the secondary button inside the expander look like a bold underlined link */
    div[data-testid="stExpander"] div[data-testid="stButton"] > button[kind="secondary"] {
        background: transparent !important;
        color: #1f6feb !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        text-decoration: underline !important;
        font-weight: 600 !important;
        cursor: pointer !important;
    }
    div[data-testid="stExpander"] { background-color: var(--secondaryBackgroundColor); border: 1px solid #CBD5E1; border-radius: 8px; margin-top: 12px; }
    div[data-testid="stExpander"] > details > summary { background-color: var(--secondaryBackgroundColor); padding: 10px 14px; border-radius: 8px 8px 0 0; }
    div[data-testid="stExpander"] > details > div[role="group"] { background-color: var(--secondaryBackgroundColor); padding: 14px; border-top: 1px solid #CBD5E1; border-radius: 0 0 8px 8px; }
    div[data-testid="stExpander"] summary p { color: #3A4556 !important; font-weight: 600; margin: 0; }
    .disclaimer-text { font-size: 15px; color: #3A4556; margin-top: 6px; line-height: 1.5; }
    @media (max-width: 600px) { button[kind="primary"] { width: 100% !important; } .disclaimer-text { font-size: 14px !important; } }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.get("accepted_disclaimer", False):
        return True

    st.session_state.setdefault("show_full_terms", False)

    def _accept():
        st.session_state["accepted_disclaimer"] = True
        st.session_state["_scroll_to_form_top_once"] = True

    def _show_full():
        st.session_state["show_full_terms"] = True

    with st.expander("Terms of Use", expanded=True):
        text_to_show = disclaimer_full if st.session_state["show_full_terms"] else disclaimer_brief
        st.markdown(text_to_show)
        if not st.session_state["show_full_terms"]:
            st.button("Read Full Terms of Use", key="btn_read_full", on_click=_show_full)
        st.button("I Accept and Proceed", key="btn_accept", type="primary", on_click=_accept)

    st.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888; margin-top: 2em;'>
            © 2025 SaluSCORE™ Project Team. All rights reserved.<br>
            Contact: <a href="mailto:isl.moh.ali@gmail.com">isl.moh.ali@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.get("accepted_disclaimer", False):
        st.stop()
    return True


def _num_step(decimals):
    try:
        d = int(decimals)
    except Exception:
        d = 1
    return 10 ** (-d) if d > 0 else 1.0


def build_user_form(schema: dict):
    import pandas as pd

    show_missing_prompt = st.session_state.get("show_missing_prompt", False)

    fields = schema.get("fields", [])
    user_fields = [f for f in fields if f.get("source") == "user" and f.get("name") != "surgery"]

    inputs = {}
    invalid_fields = []

    with st.form("patient_input", clear_on_submit=False):
        inputs = {}

        # 1) Render all user fields EXCEPT "surgery" first
        for f in user_fields:
            name  = f.get("name")
            label = f.get("label", name)
            ftype = f.get("type", "text")
            widget = f.get("widget", None)

            if ftype == "category":
                options = f.get("options", [])
                index = 0 if options else None
                if widget == "radio":
                    val = st.radio(label, options, index=index, horizontal=True, key=f"fld_{name}")
                else:
                    val = st.selectbox(label, options, index=index, key=f"fld_{name}")
                if name in ("gender", "downs") and isinstance(val, str) and val.strip().lower() == "unknown":
                    inputs[name] = None
                else:
                    inputs[name] = val

            elif ftype in ("number", "integer"):
                raw = st.text_input(label, value="", key=f"fld_{name}").strip()
                if raw == "":
                    inputs[name] = None
                else:
                    try:
                        val = int(raw) if ftype == "integer" else float(raw)
                        lo = f.get("min", None); hi = f.get("max", None)
                        if lo is not None and val < float(lo):
                            st.error(f"{label}: must be ≥ {lo}."); inputs[name] = None; invalid_fields.append(label)
                        elif hi is not None and val > float(hi):
                            st.error(f"{label}: must be ≤ {hi}."); inputs[name] = None; invalid_fields.append(label)
                        else:
                            inputs[name] = val
                    except ValueError:
                        st.error(f"Invalid input for {label}. Please enter a number.")
                        inputs[name] = None; invalid_fields.append(label)

        # 2) Planned Surgery LAST
        surgeries_selected = []
        surgery_field = next((f for f in fields if f.get("name") == "surgery"), None)
        if surgery_field:
            st.subheader(surgery_field.get("label", "Planned Surgery"))
            st.markdown(
                """
                <div style="font-size: 0.9em; color: #555;">
                Choose all relevant surgical procedures from the categories below. Only procedures included in the study are listed.
                Since this pilot study included patients with surgical VSD closure, as an isolated procedure or alongside more complex surgeries, please ensure you select a VSD closure option.
                <br><br>
                </div>
                """,
                unsafe_allow_html=True
            )
            for grp in surgery_field.get("groups", []):
                with st.expander(grp.get("title", "Group"), expanded=False):
                    options = grp.get("options", [])
                    if options and isinstance(options[0], list):
                        labels = [label for key, label in options]
                        keys   = [key   for key, label in options]
                        chosen_labels = st.multiselect("Select any that apply", labels, key=f"surg_{grp.get('title','grp')}")
                        for lab in chosen_labels:
                            idx = labels.index(lab); surgeries_selected.append(keys[idx])
                    else:
                        chosen = st.multiselect("Select any that apply", options, key=f"surg_{grp.get('title','grp')}")
                        surgeries_selected.extend(chosen)

        # 3) Submit controls
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("Analyze", type="primary", key="btn_analyze")
        with c2:
            pass
        loading_below = st.empty()

    inputs["surgery"] = surgeries_selected

    if not surgeries_selected:
        if submitted or show_missing_prompt:
            st.error("Please select at least one planned surgery to proceed.")
        st.session_state.pop("show_missing_prompt", None)
        return None

    rule = schema.get("validation", {})
    required_any = rule.get("require_any_surgery_in", [])
    if required_any and not any(s in surgeries_selected for s in required_any):
        if submitted or show_missing_prompt:
            st.error(rule.get("error_message", "Please select a required surgery option."))
        st.session_state.pop("show_missing_prompt", None)
        return None

    if invalid_fields:
        if submitted or show_missing_prompt:
            st.error("Please correct these fields before predicting: " + ", ".join(invalid_fields))
        st.session_state.pop("show_missing_prompt", None)
        return None

    # Compute missing required (excluding surgery)
    missing_required = [
        f.get("label", f.get("name"))
        for f in fields
        if (
            f.get("source") == "user"
            and f.get("required")
            and f.get("name") != "surgery"
            and (inputs.get(f.get("name")) is None or inputs.get(f.get("name")) == "")
        )
    ]

    prompt_box = st.empty()
    btn_box    = st.empty()
    intent = None

    if submitted and missing_required:
        with prompt_box:
            missing_bullets = "\n".join(f"- {x}" for x in missing_required)
            st.error("Please fill the missing fields, or click the button below to impute missing values and continue."
                     + "\n\n**Missing fields:**\n" + missing_bullets)
        with btn_box:
            if st.button("Analyze with Missing Values Filled", key="btn_impute_now", type="primary"):
                intent = "analyze_with_imputation"
                prompt_box.empty(); btn_box.empty(); loading_below.info("Loading results below…")
            else:
                st.session_state["show_missing_prompt"] = True
                return None
    elif show_missing_prompt and missing_required and not submitted:
        with prompt_box:
            missing_bullets = "\n".join(f"- {x}" for x in missing_required)
            st.error("Please fill the missing fields, or click the button below to impute missing values and continue."
                     + "\n\n**Missing fields:**\n" + missing_bullets)
        with btn_box:
            if st.button("Analyze with Missing Values Filled", key="btn_impute_now", type="primary"):
                intent = "analyze_with_imputation"
                prompt_box.empty(); btn_box.empty(); loading_below.info("Loading results below…")
            else:
                return None
    elif submitted and not missing_required:
        intent = "analyze"
        loading_below.info("Loading results below…")

    if intent is None:
        return None

    # rounding/clamping for present numeric values
    for f in [u for u in fields if u.get("source") == "user" and u.get("name") != "surgery"]:
        name = f.get("name")
        if name in inputs and isinstance(inputs[name], (float, int)):
            round_to = f.get("round_to")
            if round_to:
                inputs[name] = round(inputs[name] / round_to) * round_to
            lo = f.get("min"); hi = f.get("max")
            if lo is not None:
                inputs[name] = max(inputs[name], float(lo))
            if hi is not None:
                inputs[name] = min(inputs[name], float(hi))

    row = pd.DataFrame([inputs])
    st.session_state.pop("show_missing_prompt", None)
    return row


def render_footer():
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888; margin-top: 2em;'>
            © 2025 SaluSCORE™ Project Team. All rights reserved.<br>
            Contact: <a href="mailto:isl.moh.ali@gmail.com">isl.moh.ali@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------- PAGES -------------------------

def show_calculator():
    """Calculator page: header, Learn More button, disclaimer gate, form, prediction, SHAP, footer."""
    schema = load_schema()
    app_meta = schema.get("app", {})

    # Header
    st.title(app_meta.get("title", "SaluSCORE-PED™ v0.1"))
    st.markdown(
        "<p style='font-size:16px; color:#3A4556;'>"
        "A pilot research-only tool that uses artificial intelligence "
        "to classify in-hospital mortality risk after congenital heart "
        "surgery from preoperative data."
        "</p>",
        unsafe_allow_html=True
    )

    # Learn More (secondary) → About page
    if st.button("Learn More", type="secondary"):
        st.session_state["page"] = "about"
        st.rerun()

    # Terms (gate the calculator only)
    disclaimer_brief = (
        "This tool is a pilot prototype, provided as is, for research and educational purposes only. "
        "It must not be used for clinical decision-making or for any commercial purpose. "
        "Redistribution or modification is not permitted. By continuing, you accept full responsibility for your use of this tool."
    )
    try:
        disclaimer_full = Path(__file__).with_name("terms.md").read_text(encoding="utf-8")
    except Exception:
        disclaimer_full = "Full Terms of Use not found."
    disclaimer_gate(disclaimer_brief, disclaimer_full, owner="SaluSCORE™ Project Team", version="v0.1")

    # Jump to top once after acceptance
    if st.session_state.pop("_scroll_to_form_top_once", False):
        components.html(
            """
            <script>
            (function () {
              const d = window.parent.document;
              function jump(tries) {
                const anchor = d.querySelector('#page-top');
                if (anchor && anchor.scrollIntoView) { anchor.scrollIntoView({behavior: 'auto', block: 'start', inline: 'nearest'}); return; }
                const main = d.querySelector('section.main');
                if (main && main.scrollTo) { main.scrollTo({top: 0, left: 0, behavior: 'auto'}); return; }
                window.parent.scrollTo(0, 0);
                if (tries < 20) setTimeout(() => jump(tries + 1), 50);
              }
              setTimeout(() => jump(0), 0);
            })();
            </script>
            """,
            height=1,
            scrolling=False
        )

    # Build form
    row = build_user_form(schema)
    if row is None:
        render_footer()
        return

    # Predict
    if HAS_HELPER:
        result = predict_proba_and_shap(row, max_display=10)
        if len(result) == 4:
            proba, shap_top, traditional, fig = result
        else:
            proba, shap_top, traditional = result
            fig = None
    else:
        pipe = load_pipeline()
        proba = float(pipe.predict_proba(row)[:, 1])
        shap_top = None
        traditional = pd.Series(dtype=float)
        fig = None

    # Risk classification
    label = 1 if proba >= DECISION_THRESHOLD else 0
    label_text = "**High-risk**" if label == 1 else "**Not high-risk**"
    (st.error if label == 1 else st.success)(f"Risk classification: {label_text}")
    st.caption(f"Estimated risk probability: {proba*100:.2f}%, High-risk threshold: {DECISION_THRESHOLD*100:.2f}%")

    # Traditional Risk Scores table
    if isinstance(traditional, pd.Series) and not traditional.empty:
        st.subheader("Traditional Risk Scores")
        TRAD_THRESHOLDS = {"rachs": 3, "abc level": 3, "abc score": 7.78, "stmort category": 3, "stmort score": 0.5}
        label_map = {
            "rachs": "Risk Adjustment for Congenital Heart Surgery (RACHS-1)",
            "abc level": "Aristotle Basic Complexity (ABC) Level",
            "abc score": "Aristotle Basic Complexity (ABC) Score",
            "stmort category": "STS-EACTS Mortality Category",
            "stmort score": "STS-EACTS Mortality Score",
        }
        raw = traditional.copy()
        display_vals = {}
        for k in ("rachs", "abc level", "stmort category"):
            if k in raw and pd.notnull(raw[k]): display_vals[k] = f"{int(round(raw[k]))}"
        if "abc score" in raw and pd.notnull(raw["abc score"]): display_vals["abc score"] = f"{float(raw['abc score']):.1f}"
        if "stmort score" in raw and pd.notnull(raw["stmort score"]): display_vals["stmort score"] = f"{float(raw['stmort score']):.1f}"
        for k, v in raw.items(): display_vals.setdefault(k, v)

        rows = []
        for key, thr in TRAD_THRESHOLDS.items():
            if key in raw and pd.notnull(raw[key]):
                val = float(raw[key])
                risk = "High-risk" if val >= thr else "Not high-risk"
                rows.append({"Score": label_map.get(key, key), "Value": display_vals.get(key, val), "Risk": risk})

        trad_df = pd.DataFrame(rows, columns=["Score", "Value", "Risk"])
        trad_df = trad_df.rename(columns={"Score": "Score (Based on Planned Surgery)"})

        def risk_style(cell):
            s = str(cell).lower()
            return ("background-color:#FDECEA;color:#7A0C2E;font-weight:600;" if s == "high-risk"
                    else "background-color:#ECFDF5;color:#065F46;font-weight:600;")

        styled = trad_df.style.applymap(risk_style, subset=["Risk"]).hide(axis="index")
        st.table(styled)
        st.caption("High-risk thresholds used: RACHS-1 ≥ 3, ABC Level ≥ 3, ABC Score ≥ 7.78, "
                   "STS-EACTS Mortality Category ≥ 3, STS-EACTS Mortality Score ≥ 0.5.")

    # SHAP
    if fig is not None:
        st.subheader("Top Factors Influencing the Risk Estimate")
        st.pyplot(fig, clear_figure=True)
    elif shap_top is not None:
        st.subheader("Top Factors Influencing the Risk Estimate")
        fields = schema.get("fields", [])
        name_to_label = {f["name"]: f.get("label", f["name"]) for f in fields}
        shap_top_display = shap_top.copy()
        shap_top_display.index = [name_to_label.get(feat, feat) for feat in shap_top_display.index]
        shap_top_display = shap_top_display.fillna("")  # no "(imputed)" text
        st.dataframe(shap_top_display)

    st.divider()
    if st.button("Reset", type="primary"):
        reset_calculator_state()
        st.session_state["page"] = "calculator"  # ensure we land back on Calculator
        st.rerun()

    render_footer()


def show_about():
    """About page content and back button."""
    st.title("Learn More")

    st.markdown("""
### Why SaluSCORE? 
The name is derived from the Latin word Salus, which denotes good health. This reflects our mission to improve outcomes in congenital heart surgery by harnessing artificial intelligence responsibly.

---
### What the App Does?
SaluSCORE-PED™ is a pilot research prototype that uses preoperative demographics, labs and planned procedures to estimate in-hospital mortality risk after congenital cardiac surgery in pediatric patients.  
It currently provides a binary classification using a tuned, calibrated Bagged Extreme Gradient Boosting (XGBoost) model, with Shapley Additive Explanations to show which features most influenced the result.

---
### Study Highlights
This pilot exploratory study evaluated the model on a multicenter cohort in Egypt (566 patients), using 80% (452 patients) for training and 20% (114 patients) for internal testing, and further validated it on an external cohort (114 patients). Given the relatively small sample size, these findings should be interpreted as preliminary.
The model outperformed traditional scores, which showed area under the receiver operating characteristic curve values in the range of of 0.60-0.76, whereas the Bagged XGBoost model achieved 0.82 internally and 0.88 externally, with good calibration, Brier score 0.08.

---
### Releases
- **Release v0.1: 4 September, 2025**: Initial pilot prototype; supports binary risk classification.
    """)

    if st.button("Back to App", type="primary"):
        st.session_state["page"] = "calculator"
        st.rerun()

    render_footer()


# ------------------------- ROUTER -------------------------

def main():
    # TOP anchor (used for scroll jump after accepting terms)
    st.markdown("<div id='page-top'></div>", unsafe_allow_html=True)

    # Init page
    if "page" not in st.session_state:
        st.session_state["page"] = "calculator"

    # Route
    if st.session_state["page"] == "about":
        show_about()
    else:
        show_calculator()


if __name__ == "__main__":
    main()
