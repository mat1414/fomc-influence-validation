"""
Data loading utilities for FOMC Influence Validation Tool.
"""
import pandas as pd
import pickle
from typing import List, Dict, Optional
import streamlit as st

from config import DATA_PATHS


@st.cache_data
def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load all data files once. Cached for performance."""
    data = {}

    # Load transcripts (parquet format)
    data['transcripts'] = pd.read_parquet(DATA_PATHS["transcripts"])

    # Load pickle files
    with open(DATA_PATHS["decisions"], 'rb') as f:
        data['decisions'] = pickle.load(f)

    with open(DATA_PATHS["influence"], 'rb') as f:
        data['influence'] = pickle.load(f)

    with open(DATA_PATHS["alternatives"], 'rb') as f:
        data['alternatives'] = pickle.load(f)

    return data


def get_meeting_decisions(ymd: str, influence_df: pd.DataFrame, decisions_df: pd.DataFrame) -> pd.DataFrame:
    """Get all decisions for a meeting.

    Uses decision_id from influence data but maps to the canonical descriptions
    from adopted_decisions.pkl (which are consistent across all speakers).

    The influence data has varying descriptions for the same decision_id because
    Claude paraphrased differently for each speaker. We use the adopted_decisions
    descriptions for consistency with the alignment tool.
    """
    # Get unique decision_ids from influence data for this meeting
    meeting_influence = influence_df[influence_df['ymd'] == ymd]
    # Sort numerically, not lexicographically (so '10' comes after '9', not before '2')
    decision_ids = sorted(meeting_influence['decision_id'].unique(), key=lambda x: int(x))

    # Get canonical descriptions from adopted_decisions
    meeting_decisions = decisions_df[decisions_df['ymd'] == ymd].reset_index(drop=True)

    # Build result by mapping decision_id to adopted_decisions
    # decision_id '1', '2', '3'... maps to rows 0, 1, 2...
    rows = []
    for dec_id in decision_ids:
        idx = int(dec_id) - 1  # decision_id is 1-indexed
        if idx < len(meeting_decisions):
            row = meeting_decisions.iloc[idx]
            rows.append({
                'description': row['description'],
                'decision_id': dec_id,
                'type': row.get('type', ''),
                'score': row.get('score', 0)
            })

    return pd.DataFrame(rows)


def get_decision_speakers(ymd: str, decision_id: int, influence_df: pd.DataFrame) -> List[str]:
    """Get all speakers assessed for a specific decision.

    Uses decision_id for matching because the same decision can have
    different description text for different speakers.
    """
    mask = (influence_df['ymd'] == ymd) & (influence_df['decision_id'] == decision_id)
    speakers = influence_df[mask]['stablespeaker'].unique().tolist()
    return sorted(speakers)


def get_influence(ymd: str, decision_id: int, speaker: str, influence_df: pd.DataFrame) -> Optional[Dict]:
    """Get Claude's influence assessment for a speaker-decision pair.

    Uses decision_id for matching because the same decision can have
    different description text for different speakers.
    """
    mask = (
        (influence_df['ymd'] == ymd) &
        (influence_df['decision_id'] == decision_id) &
        (influence_df['stablespeaker'] == speaker)
    )
    rows = influence_df[mask]

    if len(rows) == 0:
        return None

    row = rows.iloc[0]
    return {
        'score': row['influence'],
        'justification': row['justification'],
        'evidence': row.get('evidence', ''),
        'first_mover': row.get('first_mover', False),
        'provided_reasoning': row.get('provided_reasoning', False),
        'cited_by_others': row.get('cited_by_others', False),
        'proposed_language': row.get('proposed_language', False),
        'discussion_extent': row.get('discussion_extent', ''),
        'who_cited_them': row.get('who_cited_them', ''),
        'speakers_before': row.get('speakers_before', '')
    }


def get_speaker_transcript(ymd: str, speaker: str, transcripts_df: pd.DataFrame) -> str:
    """Get all statements by a specific speaker in a meeting."""
    mask = (
        (transcripts_df['ymd'].astype(str) == str(ymd)) &
        (transcripts_df['stablespeaker'] == speaker)
    )
    speaker_df = transcripts_df[mask].sort_values('n')

    if len(speaker_df) == 0:
        return f"No statements found for {speaker} in this meeting."

    lines = []
    for _, row in speaker_df.iterrows():
        title = str(row.get('titletidy', '')).strip()
        name = str(row.get('stablespeaker', '')).strip()
        text = str(row.get('combined', '')).strip()

        if title and name:
            speaker_label = f"{title} {name}"
        elif name:
            speaker_label = name
        else:
            speaker_label = "UNKNOWN"

        if text:
            lines.append(f"{speaker_label}: {text}")

    return "\n\n".join(lines)


def get_full_transcript(ymd: str, transcripts_df: pd.DataFrame) -> str:
    """Get full transcript for a meeting."""
    mask = transcripts_df['ymd'].astype(str) == str(ymd)
    meeting_df = transcripts_df[mask].sort_values('n')

    if len(meeting_df) == 0:
        return f"No transcript found for meeting {ymd}."

    lines = []
    for _, row in meeting_df.iterrows():
        title = str(row.get('titletidy', '')).strip()
        name = str(row.get('stablespeaker', '')).strip()
        text = str(row.get('combined', '')).strip()

        if title and name:
            speaker_label = f"{title} {name}"
        elif name:
            speaker_label = name
        else:
            speaker_label = "UNKNOWN"

        if text:
            lines.append(f"{speaker_label}: {text}")

    return "\n\n".join(lines)


def get_alternatives(ymd: str, alternatives_df: pd.DataFrame) -> List[Dict]:
    """Get policy alternatives for a meeting."""
    meeting_alts = alternatives_df[alternatives_df['ymd'] == ymd]

    if len(meeting_alts) == 0:
        return []

    return meeting_alts.sort_values('label')[['label', 'description', 'statement']].to_dict('records')


def search_transcript(transcript_text: str, search_term: str) -> List[Dict]:
    """Search transcript for a term and return matching excerpts."""
    if not search_term:
        return []

    results = []
    search_lower = search_term.lower()

    utterances = transcript_text.split("\n\n")

    for i, utterance in enumerate(utterances):
        if search_lower in utterance.lower():
            results.append({
                "index": i,
                "text": utterance,
                "preview": utterance[:200] + "..." if len(utterance) > 200 else utterance
            })

    return results


def get_meeting_stats(ymd: str, data: Dict) -> Dict:
    """Get statistics for a meeting."""
    decisions = get_meeting_decisions(ymd, data['influence'], data['decisions'])
    influence = data['influence'][data['influence']['ymd'] == ymd]

    num_decisions = len(decisions)
    num_speakers = influence['stablespeaker'].nunique()
    num_pairs = len(influence)

    return {
        'num_decisions': num_decisions,
        'num_speakers': num_speakers,
        'num_pairs': num_pairs
    }
