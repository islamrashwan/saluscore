import json
import joblib
import pandas as pd
import streamlit as st
import datetime
import csv
from pathlib import Path
import streamlit.components.v1 as components

def disclaimer_gate(disclaimer_brief: str,
                    disclaimer_full: str,
                    owner="SaluSCORE™ Project Team",
                    version="v0.1",
                    log_to_csv=True) -> bool:
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

    # --- Styles (unchanged) ---
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
        st.session_state["_scroll_to_form_top_once"] = True  # ← add this line
        if log_to_csv:
            try:
                with open("consent_log.csv", "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([datetime.datetime.utcnow().isoformat() + "Z", owner, version])
            except Exception:
                pass

    def _show_full():
        st.session_state["show_full_terms"] = True

    with st.expander("Terms of Use", expanded=True):
        # text first
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

    fields = schema.get("fields", [])
    # render all user fields except surgery first; surgery comes later
    user_fields = [f for f in fields if f.get("source") == "user" and f.get("name") != "surgery"]

    inputs = {}

    # ----------------- FORM START -----------------
    invalid_fields = []
    with st.form("patient_input", clear_on_submit=False):
        inputs = {}

        # 1) Render all user fields EXCEPT "surgery" first (so surgery can appear at the end)
        for f in user_fields:
            name    = f.get("name")
            label   = f.get("label", name)
            ftype   = f.get("type", "text")
            widget  = f.get("widget", None)
            min_v   = f.get("min", None)
            max_v   = f.get("max", None)
            decimals= f.get("decimals", 1)

            if ftype == "category":
                options = f.get("options", [])
                index = 0 if options else None
                if widget == "radio":
                    val = st.radio(label, options, index=index, horizontal=True, key=f"fld_{name}")
                else:
                    val = st.selectbox(label, options, index=index, key=f"fld_{name}")
                inputs[name] = val

            elif ftype in ("number", "integer"):
                raw = st.text_input(
                    label,
                    value="",  # blank by default
                    key=f"fld_{name}",
                ).strip()

                if raw == "":
                    inputs[name] = None
                else:
                    try:
                        val = int(raw) if ftype == "integer" else float(raw)
                        lo = f.get("min", None)
                        hi = f.get("max", None)

                        # range check
                        if lo is not None and val < float(lo):
                            st.error(f"{label}: must be ≥ {lo}.")
                            inputs[name] = None
                            invalid_fields.append(label)
                        elif hi is not None and val > float(hi):
                            st.error(f"{label}: must be ≤ {hi}.")
                            inputs[name] = None
                            invalid_fields.append(label)
                        else:
                            inputs[name] = val

                    except ValueError:
                        st.error(f"Invalid input for {label}. Please enter a number.")
                        inputs[name] = None
                        invalid_fields.append(label)

        # 2) Planned Surgery LAST
        surgeries_selected = []
        surgery_field = next((f for f in fields if f.get("name") == "surgery"), None)
        if surgery_field:
            st.subheader(surgery_field.get("label", "Planned Surgery"))

            # Added explanatory note
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
                    # options are [["key","Label"], ...]
                    if options and isinstance(options[0], list):
                        labels = [label for key, label in options]
                        keys   = [key   for key, label in options]
                        chosen_labels = st.multiselect(
                            "Select any that apply",
                            labels,
                            key=f"surg_{grp.get('title','grp')}",
                        )
                        for lab in chosen_labels:
                            idx = labels.index(lab)
                            surgeries_selected.append(keys[idx])
                    else:
                        chosen = st.multiselect(
                            "Select any that apply",
                            options,
                            key=f"surg_{grp.get('title','grp')}",
                        )
                        surgeries_selected.extend(chosen)

        # 3) Submit controls INSIDE the form
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("Analyze", type="primary")
        with c2:
            impute_clicked = st.form_submit_button("Analyze with missing values filled", type="primary")
    # ----------------- FORM END -----------------

    # After the form rerun, read flags and handle imputation click
    confirm_impute = st.session_state.get("confirm_impute", False)
    proceed_impute = st.session_state.get("proceed_impute", False)

    if impute_clicked:
        st.session_state["confirm_impute"] = True
        st.session_state["proceed_impute"] = True
        confirm_impute = True
        proceed_impute = True

    # Consider an "action" if user clicked Predict or confirmed imputation
    action = submitted or proceed_impute
    if not action:
        return None

    # Attach surgery early so it isn't treated as missing
    inputs["surgery"] = surgeries_selected

    # ===== Priority Gates (only AFTER an action) =====

    # (A) Must have at least one planned surgery
    if not surgeries_selected:
        st.error("Please select at least one planned surgery to proceed.")
        st.session_state.pop("confirm_impute", None)
        st.session_state.pop("proceed_impute", None)
        return None

    # (B) Optional schema rule (e.g., require any of a whitelist like VSD)
    rule = schema.get("validation", {})
    required_any = rule.get("require_any_surgery_in", [])
    if required_any and not any(s in surgeries_selected for s in required_any):
        st.error(rule.get("error_message", "Please select a required surgery option."))
        st.session_state.pop("confirm_impute", None)
        st.session_state.pop("proceed_impute", None)
        return None

    # (C) Data-entry errors (letters in numeric fields, out-of-range)
    if invalid_fields:
        st.error("Please correct these fields before predicting: " + ", ".join(invalid_fields))
        st.session_state.pop("confirm_impute", None)
        st.session_state.pop("proceed_impute", None)
        return None

    # (D) Missing required (EXCLUDE surgery)
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

    # If there are missing required values and user hasn't confirmed yet → warn and stop.
    # (No buttons here: submit buttons must live inside the form.)
    if missing_required and not confirm_impute and not proceed_impute:
        st.warning(
            "The following required fields are empty and will be imputed during prediction: "
            + ", ".join(missing_required)
        )
        st.info("Please fill the missing fields or click 'Analyze with missing values filled' to impute missing values instead.")
        return None

    # If we’re here, either all required are present OR user confirmed imputation.
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

    # Build row and return (this triggers your prediction code)
    row = pd.DataFrame([inputs])

    # Clean up confirmation flags AFTER we’ve produced a row
    if st.session_state.get("confirm_impute"):
        st.session_state.pop("confirm_impute", None)
    if st.session_state.get("proceed_impute"):
        st.session_state.pop("proceed_impute", None)

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

    # ----- Header (visible while gate is open) -----
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
            // Try a few times in case layout isn't ready yet
            const d = window.parent.document;
            function jump(tries) {
                // Prefer the explicit top anchor
                const anchor = d.querySelector('#page-top');
                if (anchor && anchor.scrollIntoView) {
                anchor.scrollIntoView({behavior: 'auto', block: 'start', inline: 'nearest'});
                return;
                }
                // Fallback to the main scroll container
                const main = d.querySelector('section.main');
                if (main && main.scrollTo) { main.scrollTo({top: 0, left: 0, behavior: 'auto'}); return; }
                // Final fallback
                window.parent.scrollTo(0, 0);

                if (tries < 20) setTimeout(() => jump(tries + 1), 50);
            }
            // kick off
            setTimeout(() => jump(0), 0);
            })();
            </script>
            """,
            height=1,          # keep the iframe alive with a 1px footprint (no visible gap)
            scrolling=False
        )

    # ----- Build the form only after acceptance -----
    row = build_user_form(schema)
    if row is None:
        render_footer()   # footer before analysis
        return

    if HAS_HELPER:
        # returns: proba, shap_top, traditional, fig
        result = predict_proba_and_shap(row, max_display=10)
        # Backward-compat: older helper may return only 3 values
        if len(result) == 4:
            proba, shap_top, traditional, fig = result
        else:
            proba, shap_top, traditional = result
            fig = None
    else:
        st.warning("Running without SHAP helper (predict_and_explain.py not importable).")
        pipe = load_pipeline()
        proba = float(pipe.predict_proba(row)[:, 1])
        shap_top = None
        traditional = pd.Series(dtype=float)
        fig = None

    st.success(f"Predicted Mortality Risk: {proba*100:.2f}%")

    if not traditional.empty:
        st.subheader("Traditional Risk Scores")

        label_map = {
            "rachs": "Risk Adjustment for Congenital Heart Surgery (RACHS-1)",
            "abc level": "Aristotle Basic Complexity (ABC) Level",
            "abc score": "Aristotle Basic Complexity (ABC) Score",
            "stmort category": "STS-EACTS Mortality Category",
            "stmort score": "STS-EACTS Mortality Score"
        }

        # Format values as strings with the right precision
        formatted = traditional.copy()
        if "rachs" in formatted:
            formatted["rachs"] = f"{int(round(formatted['rachs']))}"
        if "abc level" in formatted:
            formatted["abc level"] = f"{int(round(formatted['abc level']))}"
        if "stmort category" in formatted:
            formatted["stmort category"] = f"{int(round(formatted['stmort category']))}"
        if "abc score" in formatted:
            formatted["abc score"] = f"{formatted['abc score']:.1f}"
        if "stmort score" in formatted:
            formatted["stmort score"] = f"{formatted['stmort score']:.1f}"

        trad_display = formatted.rename(index=label_map)
        st.table(trad_display.to_frame(name="Value"))

    if fig is not None:
        st.subheader("Top Feature Contributions (SHAP)")
        st.pyplot(fig, clear_figure=True)
    elif shap_top is not None:
        st.subheader("Top Feature Contributions (SHAP)")

        # Load schema and build mapping from feature name → label
        schema = load_schema()
        fields = schema.get("fields", [])
        name_to_label = {f["name"]: f.get("label", f["name"]) for f in fields}

        # Replace internal names with labels
        shap_top_display = shap_top.copy()
        shap_top_display.index = [
            name_to_label.get(feat, feat) for feat in shap_top_display.index
        ]

        # Replace missing values with "imputed"
        shap_top_display = shap_top_display.fillna("imputed")

        st.dataframe(shap_top_display)

    render_footer()  # footer after full content


if __name__ == "__main__":
    main()
