import json
import joblib
import pandas as pd
import streamlit as st
import datetime
import csv
from pathlib import Path
import streamlit.components.v1 as components

# Constants
# Probability threshold for class 1 (mortality) chosen from Youden's index
DECISION_THRESHOLD = 0.1275

def disclaimer_gate(disclaimer_brief: str,
                    disclaimer_full: str,
                    owner="SaluSCORE™ Project Team",
                    version="v0.1",
                    log_to_csv=False) -> bool:
    """
    Show brief terms first; on 'Read Full Terms of Use' swap to full text.
    Block app until 'I Accept and Proceed' is pressed, then hide permanently.
    Uses on_click callbacks (no st.rerun) for single-click behavior.
    """
    import csv, datetime
    import streamlit as st

    st.markdown("""
    <style>
    /* Make the secondary button inside the expander look like a bold underlined link */
    div[data-testid="stExpander"] div[data-testid="stButton"] > button[kind="secondary"] {
        background: transparent !important;
        color: #1f6feb !important;        /* link-ish blue */
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        text-decoration: underline !important;
        font-weight: 600 !important;       /* bold */
        cursor: pointer !important;
    }

    /* Hover/focus states */
    div[data-testid="stExpander"] div[data-testid="stButton"] > button[kind="secondary"]:hover,
    div[data-testid="stExpander"] div[data-testid="stButton"] > button[kind="secondary"]:focus {
        text-decoration: underline !important;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

    # If already accepted, let the app proceed
    if st.session_state.get("accepted_disclaimer", False):
        return True

    # Ensure flags exist
    st.session_state.setdefault("show_full_terms", False)

    # --- Card styles ---
    st.markdown("""
    <style>
    div[data-testid="stExpander"] {
        background-color: var(--secondaryBackgroundColor);
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        margin-top: 12px;
    }
    div[data-testid="stExpander"] > details > summary {
        background-color: var(--secondaryBackgroundColor);
        padding: 10px 14px;
        border-radius: 8px 8px 0 0;
    }
    div[data-testid="stExpander"] > details > div[role="group"] {
        background-color: var(--secondaryBackgroundColor);
        padding: 14px;
        border-top: 1px solid #CBD5E1;
        border-radius: 0 0 8px 8px;
    }
    div[data-testid="stExpander"] summary p {
        color: #3A4556 !important;
        font-weight: 600;
        margin: 0;
    }
    .disclaimer-text {
        font-size: 15px; 
        color: #3A4556;
        margin-top: 6px;
        line-height: 1.5;
    }
    @media (max-width: 600px) {
        button[kind="primary"] { width: 100% !important; }
        .disclaimer-text { font-size: 14px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Callbacks ---
    def _accept():
        st.session_state["accepted_disclaimer"] = True
        st.session_state["_scroll_to_form_top_once"] = True
        if log_to_csv:
            try:
                with open("consent_log.csv", "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([datetime.datetime.utcnow().isoformat() + "Z", owner, version])
            except Exception:
                pass

    def _show_full():
        st.session_state["show_full_terms"] = True

    with st.expander("Terms of Use", expanded=True):
        text_to_show = disclaimer_full if st.session_state["show_full_terms"] else disclaimer_brief
        st.markdown(text_to_show)

        # link-style secondary button (shown only on brief view)
        if not st.session_state["show_full_terms"]:
            st.button("Read Full Terms of Use", key="btn_read_full", on_click=_show_full)

        # primary accept button
        st.button("I Accept and Proceed", key="btn_accept", type="primary", on_click=_accept)

    # Footer while gated
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888; margin-top: 2em;'>
            © 2025 SaluSCORE™ Project Team. All rights reserved.<br>
            Contact: <a href="mailto:isl.moh.ali@gmail.com">isl.moh.ali@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Block the rest of the app until accepted
    if not st.session_state.get("accepted_disclaimer", False):
        st.stop()
    return True


# Try to import the SHAP + traditional score helper that returns a Matplotlib figure (no file writes)
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


def _num_step(decimals):
    try:
        d = int(decimals)
    except Exception:
        d = 1
    return 10 ** (-d) if d > 0 else 1.0


def build_user_form(schema: dict):
    import pandas as pd

    # Persisted flag: keep the prompt visible if user hasn’t fixed fields yet
    show_missing_prompt = st.session_state.get("show_missing_prompt", False)

    fields = schema.get("fields", [])
    user_fields = [f for f in fields if f.get("source") == "user" and f.get("name") != "surgery"]

    inputs = {}
    invalid_fields = []

    # ----------------- FORM START -----------------
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

        # 3) Submit controls INSIDE the form (button + a placeholder below it)
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("Analyze", type="primary", key="btn_analyze")
        with c2:
            pass
        loading_below = st.empty()  # we'll write the info box here when analysis starts
    # ----------------- FORM END -----------------

    # Attach surgery so it isn't treated as missing
    inputs["surgery"] = surgeries_selected

    # ===== Hard blockers =====
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

    # Placeholders for the prompt outside the form
    prompt_box = st.empty()
    btn_box    = st.empty()

    intent = None

    # Case 1: User clicked Analyze and there are MISSING required values
    if submitted and missing_required:
        with prompt_box:
            st.error("Please fill the missing fields, or click the button below to impute missing values and continue.")
        with btn_box:
            if st.button("Analyze with Missing Values Filled", key="btn_impute_now", type="primary"):
                intent = "analyze_with_imputation"
                # Clear the prompt area and show loading info under the Analyze button
                prompt_box.empty(); btn_box.empty()
                loading_below.info("Loading results below…")
            else:
                # keep prompt visible on next rerun; do NOT proceed
                st.session_state["show_missing_prompt"] = True
                return None

    # Case 2: Prompt already visible from a previous Analyze (user hasn’t fixed fields yet)
    elif show_missing_prompt and missing_required and not submitted:
        with prompt_box:
            st.error("Please fill the missing fields, or click the button below to impute missing values and continue.")
        with btn_box:
            if st.button("Analyze with Missing Values Filled", key="btn_impute_now", type="primary"):
                intent = "analyze_with_imputation"
                prompt_box.empty(); btn_box.empty()
                loading_below.info("Loading results below…")
            else:
                return None

    # Case 3: No missing required values and user clicked Analyze → proceed normally
    elif submitted and not missing_required:
        intent = "analyze"
        loading_below.info("Loading results below…")

    # If nothing to do yet
    if intent is None:
        return None

    # ---- rounding/clamping for present numeric values ----
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

    # Build row and return (your pipeline will impute NaNs when present)
    row = pd.DataFrame([inputs])

    # we've proceeded; no need to keep the “show prompt” flag
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


def main():
    # --- TOP anchor + injector placeholder (no visual space) ---
    st.markdown("<div id='page-top'></div>", unsafe_allow_html=True)
    _scroll_injector = st.empty()

    schema = load_schema()
    app_meta = schema.get("app", {})

    # ----- Header -----
    st.title(app_meta.get("title", "SaluSCORE-PED™ v0.1"))
    st.markdown(
        "<p style='font-size:16px; color:#3A4556;'>"
        "A pilot research-only tool that uses artificial intelligence "
        "to classify in-hospital mortality risk after congenital heart "
        "surgery from preoperative data."
        "</p>",
        unsafe_allow_html=True
    )

    # ----- Terms (brief + full) -----
    disclaimer_brief = (
        "This tool is a pilot prototype, provided as is, for research and educational purposes only. "
        "It must not be used for clinical decision-making or for any commercial purpose. "
        "Redistribution or modification is not permitted. By continuing, you accept full responsibility for your use of this tool."
    )
    disclaimer_full = Path(__file__).with_name("terms.md").read_text(encoding="utf-8")

    # Gate (blocks below until accepted)
    disclaimer_gate(disclaimer_brief, disclaimer_full, owner="SaluSCORE™ Project Team", version="v0.1")

    # After the gate returns (accepted), jump to the very top once
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

    # ----- Build the form only after acceptance -----
    row = build_user_form(schema)
    if row is None:
        render_footer()
        return

    # --- run prediction & render results ---
    if HAS_HELPER:
        # Your helper already returns calibrated probability + SHAP info
        result = predict_proba_and_shap(row, max_display=10)
        if len(result) == 4:
            proba, shap_top, traditional, fig = result
        else:
            proba, shap_top, traditional = result
            fig = None
    else:
        # Fallback: use the saved pipeline directly
        pipe = load_pipeline()
        proba = float(pipe.predict_proba(row)[:, 1])  # calibrated prob of class 1
        shap_top = None
        traditional = pd.Series(dtype=float)
        fig = None

    # ---- convert probability to class with your tuned threshold ----
    label = 1 if proba >= DECISION_THRESHOLD else 0
    label_text = "**High-risk**" if label == 1 else "**Not high-risk**"

    # Show the classification first
    if label == 1:
        st.error(f"Risk classification: {label_text}")
    else:
        st.success(f"Risk classification: {label_text}")

    # Optional transparency: show the underlying probability & threshold
    st.caption(f"Estimated risk probability: {proba*100:.2f}%, High-risk threshold: {DECISION_THRESHOLD*100:.2f}%")


    # ========= Traditional Risk Scores (with High-risk column) =========
    if isinstance(traditional, pd.Series) and not traditional.empty:
        st.subheader("Traditional Risk Scores")

        # 1) High-risk thresholds (from your Youden table)
        TRAD_THRESHOLDS = {
            "rachs": 3,
            "abc level": 3,
            "abc score": 7.78,
            "stmort category": 2.20,
            "stmort score": 0.5,
        }

        # 2) Pretty labels for the left column
        label_map = {
            "rachs": "Risk Adjustment for Congenital Heart Surgery (RACHS-1)",
            "abc level": "Aristotle Basic Complexity (ABC) Level",
            "abc score": "Aristotle Basic Complexity (ABC) Score",
            "stmort category": "STS-EACTS Mortality Category",
            "stmort score": "STS-EACTS Mortality Score",
        }

        # 3) Prepare display values (same formatting you had)
        raw = traditional.copy()
        display_vals = {}

        for k in ("rachs", "abc level", "stmort category"):
            if k in raw and pd.notnull(raw[k]):
                display_vals[k] = f"{int(round(raw[k]))}"
        if "abc score" in raw and pd.notnull(raw["abc score"]):
            display_vals["abc score"] = f"{float(raw['abc score']):.1f}"
        if "stmort score" in raw and pd.notnull(raw["stmort score"]):
            display_vals["stmort score"] = f"{float(raw['stmort score']):.1f}"
        # fall back to raw value if not formatted above
        for k, v in raw.items():
            display_vals.setdefault(k, v)

        # 4) Build: Score | Value | Risk
        rows = []
        for key, thr in TRAD_THRESHOLDS.items():
            if key in raw and pd.notnull(raw[key]):
                val = float(raw[key])
                risk = "High-risk" if val >= thr else "Not high-risk"
                rows.append({
                    "Score": label_map.get(key, key),
                    "Value": display_vals.get(key, val),
                    "Risk": risk,
                })

        trad_df = pd.DataFrame(rows, columns=["Score", "Value", "Risk"])

        # 5) Color the Risk column like Streamlit success/error
        def risk_style(cell):
            s = str(cell).lower()
            if s == "high-risk":
                # error-like (red)
                return "background-color:#FDECEA;color:#7A0C2E;font-weight:600;"
            else:
                # success-like (green)
                return "background-color:#ECFDF5;color:#065F46;font-weight:600;"

        styled = trad_df.style.applymap(risk_style, subset=["Risk"]).hide(axis="index")
        st.table(styled)

        # 6) Brief thresholds note under the table
        st.caption(
            "High-risk thresholds used: "
            "RACHS-1 ≥ 3, ABC Level ≥ 3, ABC Score ≥ 7.78, "
            "STS-EACTS Mortality Category ≥ 2.20, STS-EACTS Mortality Score ≥ 0.5."
        )



    if fig is not None:
        st.subheader("Top Feature Contributions")
        st.pyplot(fig, clear_figure=True)
    elif shap_top is not None:
        st.subheader("Top Feature Contributions")

        # Replace internal feature names with user labels from schema
        fields = schema.get("fields", [])
        name_to_label = {f["name"]: f.get("label", f["name"]) for f in fields}

        shap_top_display = shap_top.copy()
        shap_top_display.index = [
            name_to_label.get(feat, feat) for feat in shap_top_display.index
        ]
        shap_top_display = shap_top_display.fillna("imputed")

        st.dataframe(shap_top_display)

    render_footer()  # footer after full content


if __name__ == "__main__":
    main()
