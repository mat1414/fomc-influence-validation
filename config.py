# Configuration for FOMC Influence Validation Tool

TARGET_MEETINGS = {
    "19791006": "October 6, 1979 (Volcker's Saturday Night Special)",
    "19940816": "August 16, 1994 (Greenspan tightening cycle)",
    "20081216": "December 16, 2008 (Financial crisis ZLB)",
    "20110809": "August 9, 2011 (Calendar guidance introduced)",
    "20190731": "July 31, 2019 (Powell mid-cycle cut)"
}

INFLUENCE_SCALE = {
    0: "No influence",
    1: "Limited influence",
    2: "Moderate influence",
    3: "Strong influence"
}

INFLUENCE_SCALE_DETAILED = {
    0: {
        "label": "No influence",
        "criteria": "ONE OR MORE of:",
        "items": [
            "Did not mention or discuss the policy decision at all",
            "Explicitly opposed or dissented from the final decision",
            "Made statements that contradicted the final decision",
            "Their position was directly counter to what was ultimately decided"
        ]
    },
    1: {
        "label": "Limited influence",
        "criteria": "ALL of:",
        "items": [
            "Mentioned the policy decision or related topic",
            "Expressed general agreement or support",
            "Did NOT provide substantive arguments or analysis",
            "Did NOT propose specific policy parameters or language",
            "Other speakers made more detailed arguments for the same position"
        ]
    },
    2: {
        "label": "Moderate influence",
        "criteria": "AT LEAST TWO of:",
        "items": [
            "Provided explicit arguments, data, or reasoning to support the decision",
            "Discussed specific parameters or implementation details",
            "Defended the policy against alternative positions",
            "Made recommendations reflected in the final decision",
            "Raised concerns that were addressed in the final decision"
        ]
    },
    3: {
        "label": "Strong influence",
        "criteria": "AT LEAST TWO of:",
        "items": [
            "Explicitly proposed the specific policy decision that was adopted",
            "Provided unique arguments central to the committee's rationale",
            "Counterarguments shaped the final decision",
            "Suggestions appear verbatim in the final decision",
            "Was directly referenced by other speakers as shaping their views",
            "Decision would likely have been substantially different without their input"
        ]
    }
}

ACCURACY_OPTIONS = ["yes", "partially", "no"]

CONFIDENCE_LEVELS = ["high", "medium", "low"]

DATA_PATHS = {
    "transcripts": "data/transcripts.parquet",
    "decisions": "data/adopted_decisions.pkl",
    "influence": "data/policy_influence_vF.pkl",
    "alternatives": "data/alternatives.pkl",
    "results_dir": "data/coding_results/"
}
