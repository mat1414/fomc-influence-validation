"""
FOMC Influence Validation Tool

A Streamlit application for validating LLM-extracted influence scores
from Federal Reserve FOMC meeting transcripts.
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import (
    TARGET_MEETINGS,
    INFLUENCE_SCALE,
    INFLUENCE_SCALE_DETAILED,
    ACCURACY_OPTIONS,
    CONFIDENCE_LEVELS,
    DATA_PATHS
)
from utils.data_loader import (
    load_all_data,
    get_meeting_decisions,
    get_decision_speakers,
    get_influence,
    get_speaker_transcript,
    get_full_transcript,
    get_alternatives,
    search_transcript,
    get_meeting_stats
)
from utils.export import (
    generate_results_json,
    generate_results_csv,
    restore_from_uploaded_json
)


st.set_page_config(
    page_title="FOMC Influence Validation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'coder_id' not in st.session_state:
        st.session_state.coder_id = ""
    if 'selected_meeting' not in st.session_state:
        st.session_state.selected_meeting = None
    if 'current_decision_idx' not in st.session_state:
        st.session_state.current_decision_idx = 0
    if 'current_speaker_idx' not in st.session_state:
        st.session_state.current_speaker_idx = 0
    if 'validations' not in st.session_state:
        st.session_state.validations = {}
    if 'started_at' not in st.session_state:
        st.session_state.started_at = None
    if 'decisions_cache' not in st.session_state:
        st.session_state.decisions_cache = None
    if 'speakers_cache' not in st.session_state:
        st.session_state.speakers_cache = {}
    if 'jump_to_incomplete' not in st.session_state:
        st.session_state.jump_to_incomplete = False


def reset_coding_state():
    """Reset coding state when meeting changes."""
    st.session_state.current_decision_idx = 0
    st.session_state.current_speaker_idx = 0
    st.session_state.validations = {}
    st.session_state.started_at = datetime.now().isoformat()
    st.session_state.decisions_cache = None
    st.session_state.speakers_cache = {}


def get_validation_key(decision_idx: int, speaker: str) -> Tuple[int, str]:
    """Get the key for a validation entry."""
    return (decision_idx, speaker)


def get_or_create_validation(
    decision_idx: int,
    speaker: str,
    decision: Dict,
    influence_data: Dict
) -> Dict:
    """Get or create a validation entry for a speaker-decision pair."""
    key = get_validation_key(decision_idx, speaker)

    if key not in st.session_state.validations:
        st.session_state.validations[key] = {
            'decision_index': decision_idx,
            'decision_description': decision['description'],
            'decision_type': decision.get('type', ''),
            'decision_score': decision.get('score', 0),
            'speaker': speaker,
            'influence': {
                'claude_score': influence_data.get('score') if influence_data else None,
                'claude_justification': influence_data.get('justification', '') if influence_data else '',
                'claude_evidence': influence_data.get('evidence', '') if influence_data else '',
                'supports_interpretation': None,
                'human_score': None
            },
            'notes': '',
            'confidence': None,
            'completed': False,
            'completed_at': None
        }

    return st.session_state.validations[key]


def count_completed_for_decision(decision_idx: int, speakers: List[str]) -> int:
    """Count completed validations for a decision."""
    count = 0
    for speaker in speakers:
        key = get_validation_key(decision_idx, speaker)
        if key in st.session_state.validations:
            if st.session_state.validations[key].get('completed', False):
                count += 1
    return count


def count_total_completed() -> int:
    """Count total completed validations."""
    return sum(1 for v in st.session_state.validations.values() if v.get('completed', False))


def find_first_incomplete(decisions: pd.DataFrame, data: Dict) -> Tuple[int, int]:
    """Find the first incomplete speaker-decision pair."""
    ymd = st.session_state.selected_meeting

    for dec_idx, row in decisions.iterrows():
        speakers = get_decision_speakers(ymd, row['decision_id'], data['influence'])
        for sp_idx, speaker in enumerate(speakers):
            key = get_validation_key(dec_idx, speaker)
            if key not in st.session_state.validations:
                return (dec_idx, sp_idx)
            if not st.session_state.validations[key].get('completed', False):
                return (dec_idx, sp_idx)

    if len(decisions) > 0:
        last_dec = len(decisions) - 1
        last_speakers = get_decision_speakers(ymd, decisions.iloc[last_dec]['decision_id'], data['influence'])
        return (last_dec, len(last_speakers) - 1 if last_speakers else 0)

    return (0, 0)


def get_decisions_list(data: Dict) -> pd.DataFrame:
    """Get cached decisions list for current meeting."""
    if st.session_state.decisions_cache is None:
        st.session_state.decisions_cache = get_meeting_decisions(
            st.session_state.selected_meeting,
            data['influence'],
            data['decisions']
        )
    return st.session_state.decisions_cache


def get_speakers_for_decision(decision_idx: int, decision_id: int, data: Dict) -> List[str]:
    """Get cached speakers list for a decision."""
    if decision_idx not in st.session_state.speakers_cache:
        st.session_state.speakers_cache[decision_idx] = get_decision_speakers(
            st.session_state.selected_meeting,
            decision_id,
            data['influence']
        )
    return st.session_state.speakers_cache[decision_idx]


def render_sidebar(data: Dict):
    """Render the sidebar."""
    with st.sidebar:
        st.title("📈 Influence Validation")

        st.header("Coder ID")
        coder_id = st.text_input(
            "Enter your ID",
            value=st.session_state.coder_id,
            placeholder="Your initials",
            label_visibility="collapsed"
        )
        if coder_id != st.session_state.coder_id:
            st.session_state.coder_id = coder_id

        st.divider()

        st.header("Resume Previous Work")
        uploaded_file = st.file_uploader(
            "Upload saved progress (JSON)",
            type=['json'],
            help="Upload a previously downloaded JSON file to continue"
        )

        if uploaded_file is not None:
            if st.button("Restore Progress", use_container_width=True):
                success, message, restored_data = restore_from_uploaded_json(uploaded_file)
                if success and restored_data:
                    st.session_state.coder_id = restored_data['coder_id']
                    st.session_state.selected_meeting = restored_data['meeting_date']
                    st.session_state.started_at = restored_data['started_at']
                    st.session_state.validations = restored_data['validations']
                    st.session_state.decisions_cache = None
                    st.session_state.speakers_cache = {}
                    st.session_state.jump_to_incomplete = True
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.divider()

        st.header("Meeting")
        meeting_options = list(TARGET_MEETINGS.values())
        meeting_keys = list(TARGET_MEETINGS.keys())

        current_idx = None
        if st.session_state.selected_meeting in meeting_keys:
            current_idx = meeting_keys.index(st.session_state.selected_meeting)

        selected_display = st.selectbox(
            "Select meeting",
            options=meeting_options,
            index=current_idx,
            placeholder="Choose a meeting...",
            label_visibility="collapsed"
        )

        if selected_display:
            selected_ymd = meeting_keys[meeting_options.index(selected_display)]
            if selected_ymd != st.session_state.selected_meeting:
                st.session_state.selected_meeting = selected_ymd
                reset_coding_state()
                st.rerun()

        if st.session_state.selected_meeting:
            st.divider()
            st.header("Progress")

            decisions = get_decisions_list(data)
            stats = get_meeting_stats(st.session_state.selected_meeting, data)

            total_pairs = stats['num_pairs']
            completed_pairs = count_total_completed()

            st.write(f"**Total pairs:** {total_pairs}")
            st.write(f"**Completed:** {completed_pairs} ({100*completed_pairs//total_pairs if total_pairs > 0 else 0}%)")

            if total_pairs > 0:
                st.progress(completed_pairs / total_pairs)

            if completed_pairs < total_pairs:
                if st.button("Jump to Next Incomplete", use_container_width=True):
                    dec_idx, sp_idx = find_first_incomplete(decisions, data)
                    st.session_state.current_decision_idx = dec_idx
                    st.session_state.current_speaker_idx = sp_idx
                    st.rerun()

            current_dec = st.session_state.current_decision_idx
            st.write(f"**Current decision:** {current_dec + 1} of {len(decisions)}")

            if len(decisions) > 0:
                dec_row = decisions.iloc[current_dec]
                speakers = get_speakers_for_decision(current_dec, dec_row['decision_id'], data)
                speakers_done = count_completed_for_decision(current_dec, speakers)
                st.write(f"**Speakers done:** {speakers_done} of {len(speakers)}")

            st.divider()

            st.header("Decisions")

            for idx, row in decisions.iterrows():
                speakers = get_speakers_for_decision(idx, row['decision_id'], data)
                done = count_completed_for_decision(idx, speakers)
                total = len(speakers)

                if done == total and total > 0:
                    icon = "+"
                elif idx == st.session_state.current_decision_idx:
                    icon = ">"
                elif done > 0:
                    icon = "~"
                else:
                    icon = " "

                desc_short = row['description'][:40] + "..." if len(row['description']) > 40 else row['description']
                label = f"[{icon}] {idx + 1}. {desc_short}"

                if st.button(label, key=f"dec_btn_{idx}", use_container_width=True):
                    st.session_state.current_decision_idx = idx
                    st.session_state.current_speaker_idx = 0
                    st.rerun()

            st.divider()

            st.header("Download Results")

            if st.session_state.coder_id and len(st.session_state.validations) > 0:
                decisions_list = [
                    {'description': row['description'], 'type': row.get('type', ''), 'score': row.get('score', 0)}
                    for _, row in decisions.iterrows()
                ]

                json_data, json_filename = generate_results_json(
                    st.session_state.selected_meeting,
                    st.session_state.coder_id,
                    st.session_state.validations,
                    decisions_list,
                    st.session_state.started_at or datetime.now().isoformat()
                )

                csv_data, csv_filename = generate_results_csv(
                    st.session_state.selected_meeting,
                    st.session_state.coder_id,
                    st.session_state.validations
                )

                st.download_button(
                    "Download JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )

                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Complete some validations to enable downloads")


def render_decision_context(decision: Dict, ymd: str, data: Dict):
    """Render the decision context section."""
    with st.expander("Decision Context", expanded=True):
        st.markdown(f"**Decision {st.session_state.current_decision_idx + 1}**")
        st.info(decision['description'])

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Type:** {decision.get('type', 'N/A')}")
        with col2:
            score = decision.get('score', 0)
            st.write(f"**Policy stance:** {score}")

        alternatives = get_alternatives(ymd, data['alternatives'])
        if alternatives:
            with st.expander("View Policy Alternatives"):
                for alt in alternatives:
                    st.markdown(f"**{alt['label']}**")
                    st.caption(alt['description'])
                    st.text_area(
                        f"Statement - {alt['label']}",
                        value=alt['statement'],
                        height=100,
                        disabled=True,
                        label_visibility="collapsed"
                    )
        else:
            st.caption("No policy alternatives available for this meeting.")


def render_speaker_navigation(speakers: List[str], decision_idx: int):
    """Render speaker navigation controls."""
    current_idx = st.session_state.current_speaker_idx
    current_speaker = speakers[current_idx] if current_idx < len(speakers) else speakers[0]

    st.markdown("---")
    st.subheader(f"Speaker {current_idx + 1} of {len(speakers)}: {current_speaker}")

    progress_str = ""
    for i, sp in enumerate(speakers):
        key = get_validation_key(decision_idx, sp)
        if key in st.session_state.validations and st.session_state.validations[key].get('completed'):
            progress_str += "[+] "
        elif i == current_idx:
            progress_str += "[>] "
        else:
            progress_str += "[ ] "
    st.caption(f"Progress: {progress_str}")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("< Previous", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.current_speaker_idx = current_idx - 1
            st.rerun()

    with col3:
        if st.button("Next >", disabled=(current_idx >= len(speakers) - 1), use_container_width=True):
            st.session_state.current_speaker_idx = current_idx + 1
            st.rerun()

    return current_speaker


def render_assessment_form(
    speaker: str,
    decision: Dict,
    decision_idx: int,
    influence_data: Dict,
    speakers: List[str],
    decisions: pd.DataFrame,
    data: Dict
):
    """Render the influence assessment form."""
    validation = get_or_create_validation(
        decision_idx, speaker, decision, influence_data
    )

    st.markdown("---")
    st.markdown("### Influence Assessment")

    inf = validation['influence']
    claude_inf_score = inf.get('claude_score')

    if claude_inf_score is not None:
        score_label = INFLUENCE_SCALE.get(int(claude_inf_score), "Unknown")
        st.metric("Claude's score", f"{claude_inf_score} ({score_label})")
    else:
        st.warning("No influence data available")

    st.markdown("**Justification:**")
    st.text_area(
        "Influence justification",
        value=inf.get('claude_justification', 'No justification provided'),
        height=150,
        disabled=True,
        key=f"inf_just_{decision_idx}_{speaker}",
        label_visibility="collapsed"
    )

    # Show evidence if available
    evidence = inf.get('claude_evidence', '')
    if evidence:
        st.markdown("**Evidence cited:**")
        st.text_area(
            "Evidence",
            value=evidence,
            height=100,
            disabled=True,
            key=f"inf_evidence_{decision_idx}_{speaker}",
            label_visibility="collapsed"
        )

    st.markdown("---")
    st.markdown("**Your Assessment:**")

    st.markdown("1. Evidence supports score?")
    supports_interp = st.radio(
        "Supports interpretation",
        options=["Yes", "Partially", "No"],
        index=["yes", "partially", "no"].index(inf.get('supports_interpretation')) if inf.get('supports_interpretation') in ["yes", "partially", "no"] else None,
        key=f"supports_inf_{decision_idx}_{speaker}",
        horizontal=True,
        label_visibility="collapsed"
    )
    if supports_interp:
        inf['supports_interpretation'] = supports_interp.lower()

    st.markdown("2. Your influence score:")

    with st.expander("Scale reference", expanded=False):
        for score, details in INFLUENCE_SCALE_DETAILED.items():
            st.markdown(f"**{score}: {details['label']}** ({details['criteria']})")
            for item in details['items']:
                st.caption(f"  - {item}")

    human_inf_score = st.radio(
        "Your influence score",
        options=[0, 1, 2, 3],
        index=inf.get('human_score') if inf.get('human_score') is not None else 0,
        format_func=lambda x: f"{x} - {INFLUENCE_SCALE[x]}",
        key=f"human_inf_{decision_idx}_{speaker}",
        horizontal=True,
        label_visibility="collapsed"
    )
    inf['human_score'] = human_inf_score

    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Notes (optional):**")
        notes = st.text_area(
            "Notes",
            value=validation.get('notes', ''),
            key=f"notes_{decision_idx}_{speaker}",
            placeholder="Any observations, concerns, or comments...",
            label_visibility="collapsed"
        )
        validation['notes'] = notes

    with col2:
        st.markdown("**Confidence:**")
        confidence = st.radio(
            "Confidence level",
            options=["High", "Medium", "Low"],
            index=["high", "medium", "low"].index(validation.get('confidence')) if validation.get('confidence') in ["high", "medium", "low"] else None,
            key=f"confidence_{decision_idx}_{speaker}",
            label_visibility="collapsed"
        )
        if confidence:
            validation['confidence'] = confidence.lower()

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Submit & Next", type="primary", use_container_width=True):
            missing = []
            if inf.get('supports_interpretation') is None:
                missing.append("Evidence supports score")
            if validation.get('confidence') is None:
                missing.append("Confidence")

            if missing:
                st.error(f"Please complete: {', '.join(missing)}")
            else:
                validation['completed'] = True
                validation['completed_at'] = datetime.now().isoformat()

                key = get_validation_key(decision_idx, speaker)
                st.session_state.validations[key] = validation

                current_speaker_idx = st.session_state.current_speaker_idx

                if current_speaker_idx < len(speakers) - 1:
                    st.session_state.current_speaker_idx = current_speaker_idx + 1
                elif st.session_state.current_decision_idx < len(decisions) - 1:
                    st.session_state.current_decision_idx += 1
                    st.session_state.current_speaker_idx = 0
                else:
                    st.success("All validations complete!")
                    st.balloons()

                st.rerun()


def render_transcript_section(ymd: str, speaker: str, data: Dict):
    """Render the transcript viewer section."""
    with st.expander("Transcript", expanded=False):
        view_mode = st.radio(
            "View mode",
            options=[f"This speaker only ({speaker})", "Full transcript"],
            horizontal=True,
            label_visibility="collapsed"
        )

        search_term = st.text_input("Search transcript", placeholder="Enter search term...")

        if "This speaker only" in view_mode:
            transcript = get_speaker_transcript(ymd, speaker, data['transcripts'])
        else:
            transcript = get_full_transcript(ymd, data['transcripts'])

        if search_term:
            results = search_transcript(transcript, search_term)
            st.write(f"Found {len(results)} matches")
            for result in results[:50]:
                highlighted = result['text'].replace(search_term, f"**{search_term}**")
                st.markdown(f"---\n{highlighted}")
        else:
            st.text_area(
                "Transcript content",
                value=transcript,
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )


def main():
    """Main application entry point."""
    init_session_state()

    data = load_all_data()

    render_sidebar(data)

    if not st.session_state.coder_id:
        st.warning("Please enter your Coder ID in the sidebar to begin.")
        st.stop()

    if not st.session_state.selected_meeting:
        st.info("Please select a meeting from the sidebar to begin validation.")

        st.markdown("### Available Meetings")
        for ymd, name in TARGET_MEETINGS.items():
            stats = get_meeting_stats(ymd, data)
            st.markdown(f"- **{name}**: {stats['num_decisions']} decisions, {stats['num_speakers']} speakers, {stats['num_pairs']} pairs")
        st.stop()

    ymd = st.session_state.selected_meeting
    decisions = get_decisions_list(data)

    if len(decisions) == 0:
        st.error("No decisions found for this meeting.")
        st.stop()

    if st.session_state.jump_to_incomplete:
        st.session_state.jump_to_incomplete = False
        dec_idx, sp_idx = find_first_incomplete(decisions, data)
        st.session_state.current_decision_idx = dec_idx
        st.session_state.current_speaker_idx = sp_idx
        completed = count_total_completed()
        total = get_meeting_stats(ymd, data)['num_pairs']
        st.toast(f"Restored! {completed}/{total} pairs completed. Jumping to next incomplete.")
        st.rerun()

    decision_idx = st.session_state.current_decision_idx
    decision_row = decisions.iloc[decision_idx]
    decision = {
        'description': decision_row['description'],
        'decision_id': decision_row['decision_id'],
        'type': decision_row.get('type', ''),
        'score': decision_row.get('score', 0)
    }

    speakers = get_speakers_for_decision(decision_idx, decision['decision_id'], data)

    if len(speakers) == 0:
        st.error("No speakers found for this decision.")
        st.stop()

    if st.session_state.current_speaker_idx >= len(speakers):
        st.session_state.current_speaker_idx = 0

    st.title("FOMC Influence Validation")

    render_decision_context(decision, ymd, data)

    current_speaker = render_speaker_navigation(speakers, decision_idx)

    render_transcript_section(ymd, current_speaker, data)

    influence_data = get_influence(ymd, decision['decision_id'], current_speaker, data['influence'])

    render_assessment_form(
        current_speaker,
        decision,
        decision_idx,
        influence_data,
        speakers,
        decisions,
        data
    )


if __name__ == "__main__":
    main()
