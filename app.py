from pathlib import Path
import streamlit as st

# Tab title + favicon
st.set_page_config(
    page_title="SaluSCORE",          # text in the browser tab
    page_icon=str(Path(__file__).with_name("favicon.png")), # PNG/ICO/SVG in your repo
    layout="centered"
)


import json
import joblib
import pandas as pd
import streamlit.components.v1 as components


# --- Preflight: ensure stdlib pathlib is used (not a backport or local package) ---
import pathlib, sys
_pf = getattr(pathlib, "__file__", "") or ""
if "site-packages" in _pf.lower():
    raise RuntimeError(f"Bad pathlib found at: {_pf}. Remove any 'pathlib' wheels/backports or local modules.")

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


def disclaimer_gate(disclaimer_brief: str,
                    disclaimer_full: str,
                    owner="SaluSCORE™ Project Team",
                    version="v0.1",
                    log_to_csv=False) -> bool:
    """
    Show brief terms in an expander and block the app until
    'I Accept and Proceed' is pressed. Once accepted, hide permanently.
    """
    # If already accepted, do nothing
    if st.session_state.get("accepted_disclaimer", False):
        return True

    def _accept():
        st.session_state["accepted_disclaimer"] = True
        st.session_state["_scroll_to_form_top_once"] = True

    with st.expander("Terms of Use", expanded=True):
        # Only show the brief text now (no "Read full terms" link)
        st.markdown(disclaimer_brief)
        st.button("I Accept and Proceed", key="btn_accept", type="primary", on_click=_accept)

    # Small footer under the terms box
    st.markdown(
        f"""
        <div style='text-align: center; font-size: 0.8em; color: #888; margin-top: 2em;'>
            © 2025 {owner}. All rights reserved. <br>
            Contact: <a href="mailto:isl.moh.ali@gmail.com">isl.moh.ali@gmail.com</a><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Block the rest of the app until accepted
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

            # --- Mark obligatory fields with "*" ---
            OBLIGATORY_LABELS = {
                "Gender",
                "Age (years)",
                "Down Syndrome",
                "Hemoglobin (g/dL)",
                "Hematocrit (%)",
                "Mean Corpuscular Volume (fL)",
                "Mean Corpuscular Hemoglobin (pg)",
                "Mean Corpuscular Hemoglobin Concentration (g/dL)",
                "Platelets (×10^9/L)",
                "Total Leucocyte Count (×10^9/L)",
            }
            if label in OBLIGATORY_LABELS:
                label = f"{label} *"

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

            # compact spacing for expanders / blocks (applies globally)
            st.markdown("""
            <style>
            /* shrink the default vertical gap between Streamlit blocks */
            div[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }

            /* tighten expander spacing and paddings */
            div[data-testid="stExpander"] { margin: 4px 0 !important; }
            div[data-testid="stExpander"] > details > summary { padding: 8px 12px !important; }
            div[data-testid="stExpander"] > details > div[role="group"] { padding: 8px 12px !important; }
            </style>
            """, unsafe_allow_html=True)

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
            submitted = st.form_submit_button("Analyze", type="primary")
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

    # --- NEW: separate obligatory vs optional-imputable fields ---
    obligatory_names = {
        "gender", "age", "downs", "hb", "hct", "mcv", "mch", "mchc", "plt", "tlc"
    }
    optional_imputable_names = {
        "weight", "spo2", "inr", "ptt", "creat", "urea", "alt", "ast"
    }

    # helper to map internal names to labels from the schema
    name_to_label = {f.get("name"): f.get("label", f.get("name")) for f in fields}

    # what's missing?
    missing_oblig = [
        name_to_label[n] for n in obligatory_names
        if (n not in inputs) or (inputs.get(n) is None) or (inputs.get(n) == "")
    ]
    missing_optional = [
        name_to_label[n] for n in optional_imputable_names
        if (n not in inputs) or (inputs.get(n) is None) or (inputs.get(n) == "")
    ]

    prompt_box = st.empty()
    btn_box    = st.empty()
    intent = None

    # 1) If any obligatory fields are missing -> STOP here (no imputation offered)
    if submitted and missing_oblig:
        with prompt_box:
            missing_bullets = "\n".join(f"- {x}" for x in sorted(missing_oblig))
            st.error(
                "Please fill the **obligatory** fields before continuing."
                + "\n\n**Obligatory fields missing:**\n" + missing_bullets
            )
        st.session_state.pop("show_missing_prompt", None)
        return None

    # 2) If obligatory are present but optional are missing -> offer imputation
    if submitted and not missing_oblig and missing_optional:
        with prompt_box:
            missing_bullets = "\n".join(f"- {x}" for x in sorted(missing_optional))
            st.warning(
                "Some optional fields are missing. You can either fill them or proceed "
                "with **imputation** for the missing optional values."
                + "\n\n**Optional fields missing:**\n" + missing_bullets
            )
        with btn_box:
            if st.button("Analyze with Missing Optional Values Imputed", key="btn_impute_now", type="primary"):
                intent = "analyze_with_imputation"
                prompt_box.empty(); btn_box.empty(); loading_below.info("Loading results below…")
            else:
                st.session_state["show_missing_prompt"] = True
                return None

    # 3) Handle the reminder case when the session asked to show the prompt again
    elif show_missing_prompt and not submitted and not missing_oblig and missing_optional:
        with prompt_box:
            missing_bullets = "\n".join(f"- {x}" for x in sorted(missing_optional))
            st.warning(
                "Some optional fields are still missing. You may proceed with **imputation** "
                "or fill them first."
                + "\n\n**Optional fields missing:**\n" + missing_bullets
            )
        with btn_box:
            if st.button("Analyze with Missing Optional Values Imputed", key="btn_impute_now", type="primary"):
                intent = "analyze_with_imputation"
                prompt_box.empty(); btn_box.empty(); loading_below.info("Loading results below…")
            else:
                return None

    # 4) If nothing missing -> normal analyze
    elif submitted and not missing_oblig and not missing_optional:
        intent = "analyze"
        loading_below.info("Loading results below…")

    # Build the single-row dataframe when an action is chosen
    if intent in ("analyze", "analyze_with_imputation"):
        # 'inputs' already contains all user fields + 'surgery'
        row = pd.DataFrame([inputs])
        # (optional) remember which path was chosen
        st.session_state["analysis_intent"] = intent
        return row


    st.markdown("<small>**\\*** Obligatory field — must be filled before analysis</small>", unsafe_allow_html=True)


    # no action yet → keep building UI
    return None


def render_footer():
    # About / summary section
    st.markdown(
        """
        ---
        ### About SaluSCORE

        **Why SaluSCORE?**  
        The name is derived from the Latin word Salus, which denotes good health. This reflects our mission to improve outcomes in congenital heart surgery by harnessing artificial intelligence responsibly.

        **What the App Does**  
        SaluSCORE-PED™ is a pilot research prototype that uses preoperative demographics, labs and planned procedures to estimate in-hospital mortality risk after congenital cardiac surgery in pediatric patients.  
        It currently provides a binary classification using a tuned, calibrated Bagged Extreme Gradient Boosting (XGBoost) model, with Shapley Additive Explanations to show which features most influenced the result.

        **Study Highlights**  
        This pilot exploratory study evaluated the model on a multicenter cohort in Egypt (566 patients), using 80% (452 patients) for training and 20% (114 patients) for internal testing, and further validated it on an external cohort (114 patients). Given the relatively small sample size, these findings should be interpreted as preliminary.
        The model outperformed traditional scores, which showed area under the receiver operating characteristic curve values in the range of of 0.60-0.76, whereas the Bagged XGBoost model achieved 0.82 internally and 0.88 externally, with good calibration, Brier score 0.08.

        **Release v0.1 — 4 September 2025**  
        Initial pilot prototype; supports binary risk classification.
        """,
        unsafe_allow_html=False,
    )

    # Copyright / contact
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888; margin-top: 2em;'>
            © 2025 SaluSCORE™ Project Team. All rights reserved.<br>
            Contact: <a href="mailto:isl.moh.ali@gmail.com">isl.moh.ali@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_results(proba, shap_top, traditional, fig, schema):
    import pandas as pd
    DECISION_THRESHOLD = 0.1275

    # Risk classification
    label = 1 if proba >= DECISION_THRESHOLD else 0
    label_text = "**High-risk**" if label == 1 else "**Not high-risk**"
    (st.error if label == 1 else st.success)(f"Risk classification: {label_text}")
    st.caption(f"Estimated risk probability: {proba*100:.2f}%, High-risk threshold: {DECISION_THRESHOLD*100:.2f}%.")

    # Traditional Risk Scores (table with risk highlighting)
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

        trad_df = pd.DataFrame(rows, columns=["Score", "Value", "Risk"]).rename(
            columns={"Score": "Score (Based on Planned Surgery)"}
        )

        def risk_style(cell):
            s = str(cell).lower()
            return ("background-color:#FDECEA;color:#7A0C2E;font-weight:600;" if s == "high-risk"
                    else "background-color:#ECFDF5;color:#065F46;font-weight:600;")

        styled = trad_df.style.applymap(risk_style, subset=["Risk"]).hide(axis="index")
        st.table(styled)
        st.caption("High-risk thresholds used: RACHS-1 ≥ 3, ABC Level ≥ 3, ABC Score ≥ 7.78, "
                   "STS-EACTS Mortality Category ≥ 3, STS-EACTS Mortality Score ≥ 0.5.")

    # SHAP figure or (fallback) table
    if fig is not None:
        st.subheader("Top Factors Influencing the Risk Estimate")
        st.pyplot(fig, clear_figure=True)
        st.markdown(
            "Red bars push the prediction towards higher risk. Blue bars push the prediction towards lower risk. "
            "The longer the bar, the stronger the effect. Feature labels indicate the patient's input values."
        )
    elif shap_top is not None:
        st.subheader("Top Factors Influencing the Risk Estimate")
        fields = schema.get("fields", [])
        name_to_label = {f["name"]: f.get("label", f["name"]) for f in fields}
        shap_top_display = shap_top.copy()
        shap_top_display.index = [name_to_label.get(feat, feat) for feat in shap_top_display.index]
        shap_top_display = shap_top_display.fillna("")
        st.dataframe(shap_top_display)



# ------------------------- PAGES -------------------------

def show_calculator():
    """Calculator page: header, disclaimer gate, form, prediction, SHAP, footer."""
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


    # --- Results-only view: if results are in session, skip the form (hides Analyze buttons) ---
    if st.session_state.get("analysis_done") and "result_payload" in st.session_state:
        proba, shap_top, traditional, fig = st.session_state["result_payload"]
        render_results(proba, shap_top, traditional, fig, schema)

        if st.button("New Analysis", type="primary"):
            st.session_state.pop("analysis_done", None)
            st.session_state.pop("result_payload", None)
            st.rerun()

        render_footer()
        return


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

    # Save to session and rerun into results-only view
    st.session_state["result_payload"] = (proba, shap_top, traditional, fig)
    st.session_state["analysis_done"] = True
    st.rerun()


# ------------------------- ROUTER -------------------------

def main():
    # TOP anchor (used for scroll jump after accepting terms)
    st.markdown("<div id='page-top'></div>", unsafe_allow_html=True)

    # Just show the calculator page
    show_calculator()


if __name__ == "__main__":
    main()
