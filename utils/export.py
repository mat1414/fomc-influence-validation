"""
Export utilities for FOMC Influence Validation Tool.
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from config import TARGET_MEETINGS


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


def generate_results_json(
    ymd: str,
    coder_id: str,
    validations: Dict,
    decisions: List[Dict],
    started_at: str
) -> Tuple[str, str]:
    """
    Generate coding results as JSON string.

    Args:
        ymd: Meeting date
        coder_id: Coder identifier
        validations: Dict mapping (decision_idx, speaker) to validation data
        decisions: List of decision dicts
        started_at: ISO timestamp when coding started

    Returns:
        Tuple of (json_string, filename)
    """
    num_completed = sum(1 for v in validations.values() if v.get('completed', False))
    num_total = len(validations)

    output = {
        "metadata": {
            "meeting_date": ymd,
            "coder_id": coder_id,
            "started_at": started_at,
            "last_saved": datetime.now().isoformat(),
            "app_version": "1.0",
            "validation_type": "influence",
            "num_decisions": len(decisions),
            "num_pairs_total": num_total,
            "num_pairs_completed": num_completed
        },
        "validations": list(validations.values())
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"influence_{ymd}_{coder_id}_{timestamp}.json"

    return json.dumps(output, indent=2, ensure_ascii=False, cls=NumpyEncoder), filename


def generate_results_csv(
    ymd: str,
    coder_id: str,
    validations: Dict
) -> Tuple[str, str]:
    """
    Generate coding results as CSV string.

    Args:
        ymd: Meeting date
        coder_id: Coder identifier
        validations: Dict mapping (decision_idx, speaker) to validation data

    Returns:
        Tuple of (csv_string, filename)
    """
    rows = []

    for key, val in validations.items():
        influence = val.get('influence', {})

        row = {
            "meeting_date": ymd,
            "coder_id": coder_id,
            "decision_index": val.get("decision_index"),
            "decision_description": val.get("decision_description"),
            "decision_type": val.get("decision_type"),
            "decision_score": val.get("decision_score"),
            "speaker": val.get("speaker"),
            "claude_influence_score": influence.get("claude_score"),
            "claude_influence_justification": influence.get("claude_justification"),
            "claude_evidence": influence.get("claude_evidence"),
            "influence_supports_interpretation": influence.get("supports_interpretation"),
            "human_influence_score": influence.get("human_score"),
            "notes": val.get("notes", ""),
            "confidence": val.get("confidence"),
            "completed": val.get("completed", False),
            "completed_at": val.get("completed_at")
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(['decision_index', 'speaker'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"influence_{ymd}_{coder_id}_{timestamp}.csv"

    return df.to_csv(index=False), filename


def restore_from_uploaded_json(uploaded_file) -> Tuple[bool, str, Optional[Dict]]:
    """
    Restore coding progress from an uploaded JSON file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (success, message, restored_data)
    """
    try:
        content = uploaded_file.read().decode('utf-8')
        data = json.loads(content)

        metadata = data.get('metadata', {})
        meeting_date = metadata.get('meeting_date')
        coder_id = metadata.get('coder_id')
        started_at = metadata.get('started_at')

        if not meeting_date or meeting_date not in TARGET_MEETINGS:
            return False, "Invalid or unsupported meeting date in file", None

        # Check validation type
        if metadata.get('validation_type') and metadata.get('validation_type') != 'influence':
            return False, "This file contains alignment validations, not influence", None

        validations_list = data.get('validations', [])
        validations = {}

        for val in validations_list:
            decision_idx = val.get('decision_index')
            speaker = val.get('speaker')
            if decision_idx is not None and speaker:
                key = (decision_idx, speaker)
                validations[key] = val

        restored_data = {
            'meeting_date': meeting_date,
            'coder_id': coder_id,
            'started_at': started_at,
            'validations': validations
        }

        return True, f"Restored progress: {len(validations)} validations for meeting {meeting_date}", restored_data

    except json.JSONDecodeError:
        return False, "Invalid JSON file", None
    except Exception as e:
        return False, f"Error loading file: {str(e)}", None
