"""Raptors defense analysis using the methods in defense_testing.ipynb."""

from pathlib import Path

import numpy as np
import pandas as pd

RAPTORS_TEAM_ID = 1610612761


def analyze_defense_game(game_id, play_by_play, movement_data,
                         possession_directory="multi_game_data", max_link_offset_seconds=15):
    """Return per-game PBP, possession, and movement outputs plus matching/coverage audits.

    Uses the notebook's missed-shot scan, scoring scan, clock conversion, and
    chronological nearest-start matching. Out-of-tolerance defensive events are
    reported separately. Confirmed continuation and scoring labels follow the
    existing multi-game movement export. Missing tracking is never synthesized.
    """
    GAME_ID = int(game_id)
    TRACKING_GAME_ID = f"{GAME_ID:010d}"
    possession_path = Path(possession_directory) / f"{TRACKING_GAME_ID}_possessions.csv"
    game_pbp = play_by_play.loc[pd.to_numeric(play_by_play["GAME_ID"]).eq(GAME_ID)].copy()
    raptors_movement_data = movement_data.loc[
        pd.to_numeric(movement_data["game_id"]).eq(GAME_ID)
    ].copy()
    if game_pbp.empty or raptors_movement_data.empty:
        raise ValueError(f"Missing play-by-play or movement data for {TRACKING_GAME_ID}.")
    sides = pd.to_numeric(raptors_movement_data["is_home"]).unique()
    if len(sides) != 1 or sides[0] not in (0, 1):
        raise ValueError(f"Ambiguous Raptors home/away side for {TRACKING_GAME_ID}.")
    is_home = bool(sides[0])
    raptors_description_column = "HOMEDESCRIPTION" if is_home else "VISITORDESCRIPTION"
    opponent_description_column = "VISITORDESCRIPTION" if is_home else "HOMEDESCRIPTION"

    game = (
        game_pbp.sort_values(["PERIOD", "EVENTNUM"])
        .reset_index(drop=True)
        .copy()
    )
    game["EVENTMSGTYPE"] = pd.to_numeric(game["EVENTMSGTYPE"], errors="coerce")
    opponent_desc = game[opponent_description_column].fillna("").astype(str)
    raptors_desc = game[raptors_description_column].fillna("").astype(str)


    def previous_missed_shot(row_number):
        """Return the prior missed-shot row, ignoring administrative events."""
        for prior in range(row_number - 1, -1, -1):
            event_type = game.at[prior, "EVENTMSGTYPE"]
            if event_type in (2, 3):
                descriptions = f"{opponent_desc.iat[prior]} {raptors_desc.iat[prior]}"
                if "MISS" in descriptions.upper():
                    return prior
            if event_type in (1, 4, 5, 12, 13):
                break
        return None


    possession_losses = []
    for row_number, row in game.iterrows():
        event_type = row["EVENTMSGTYPE"]

        # A recorded opponent turnover, including a turnover caused by a Raptors steal.
        if event_type == 5 and "TURNOVER" in opponent_desc.iat[row_number].upper():
            causes = ["turnover"]
            if "STEAL" in raptors_desc.iat[row_number].upper():
                causes.append("steal")
            possession_losses.append(
                {
                    "start_row": row_number,
                    "PERIOD": row["PERIOD"],
                    "PCTIMESTRING": row["PCTIMESTRING"],
                    "LOSS_EVENTNUM": row["EVENTNUM"],
                    "OPPONENT_EVENT": opponent_desc.iat[row_number],
                    "RAPTORS_TAKEAWAY": raptors_desc.iat[row_number],
                    "CAUSE": " + ".join(causes),
                }
            )

        # A Raptors rebound is defensive only when it follows a missed opponent shot.
        elif event_type == 4 and raptors_desc.iat[row_number]:
            shot_row = previous_missed_shot(row_number)
            if shot_row is not None and opponent_desc.iat[shot_row].upper().startswith("MISS"):
                causes = ["defensive rebound"]
                if "BLOCK" in raptors_desc.iat[shot_row].upper():
                    causes.append("block")
                possession_losses.append(
                    {
                        "start_row": row_number,
                        "PERIOD": row["PERIOD"],
                        "PCTIMESTRING": row["PCTIMESTRING"],
                        "LOSS_EVENTNUM": row["EVENTNUM"],
                        "OPPONENT_EVENT": opponent_desc.iat[shot_row],
                        "RAPTORS_TAKEAWAY": raptors_desc.iat[row_number],
                        "CAUSE": " + ".join(causes),
                    }
                )


    def resulting_raptors_score(start_row):
        """Find a Raptors score before that Raptors possession ends."""
        for row_number in range(start_row + 1, len(game)):
            event_type = game.at[row_number, "EVENTMSGTYPE"]
            opponent_event = opponent_desc.iat[row_number]
            raptors_event = raptors_desc.iat[row_number]

            if event_type == 13:  # End of period
                return None
            if raptors_event and event_type == 1:
                return row_number, raptors_event
            if raptors_event and event_type == 3:
                # Combine all Raptors free throws taken at this clock stoppage.
                free_throw_rows = []
                for free_throw_row in range(row_number, len(game)):
                    same_period = game.at[free_throw_row, "PERIOD"] == game.at[row_number, "PERIOD"]
                    same_clock = game.at[free_throw_row, "PCTIMESTRING"] == game.at[row_number, "PCTIMESTRING"]
                    if not (same_period and same_clock):
                        break
                    free_throw_event = raptors_desc.iat[free_throw_row]
                    if game.at[free_throw_row, "EVENTMSGTYPE"] == 3 and free_throw_event:
                        free_throw_rows.append(free_throw_row)

                made_free_throws = [
                    free_throw_row
                    for free_throw_row in free_throw_rows
                    if "MISS" not in raptors_desc.iat[free_throw_row].upper()
                ]
                if made_free_throws:
                    combined_description = " | ".join(
                        raptors_desc.iat[free_throw_row] for free_throw_row in free_throw_rows
                    )
                    return made_free_throws[0], combined_description
            if event_type == 5 and raptors_event:  # Raptors turnover
                return None
            if event_type == 4 and opponent_event:  # Opponent rebound of a Raptors miss
                return None
            if event_type in (1, 5) and opponent_event:  # Opponent already regained possession
                return None
        return None


    converted_losses = []
    for loss in possession_losses:
        score = resulting_raptors_score(loss["start_row"])
        if score is not None:
            score_row, score_description = score
            converted_losses.append(
                {
                    key: value for key, value in loss.items() if key != "start_row"
                }
                | {
                    "SCORE_EVENTNUM": game.at[score_row, "EVENTNUM"],
                    "RAPTORS_SCORE": score_description,
                }
            )

    converted_opponent_possession_losses = pd.DataFrame(converted_losses, columns=[
        "PERIOD", "PCTIMESTRING", "LOSS_EVENTNUM", "OPPONENT_EVENT",
        "RAPTORS_TAKEAWAY", "CAUSE", "SCORE_EVENTNUM", "RAPTORS_SCORE",
    ])
    cause_summary = pd.Series(
        {
            "Defensive rebounds": converted_opponent_possession_losses["CAUSE"].str.contains("defensive rebound").sum(),
            "Blocks (subset of defensive rebounds)": converted_opponent_possession_losses["CAUSE"].str.contains("block").sum(),
            "Steals": converted_opponent_possession_losses["CAUSE"].str.contains("steal").sum(),
            "Other recorded turnovers": (converted_opponent_possession_losses["CAUSE"] == "turnover").sum(),
        },
        name="COUNT",
    )

    game_possessions = pd.read_csv(possession_path)
    game_possessions["offense_team_id"] = pd.to_numeric(
        game_possessions["offense_team_id"], errors="coerce"
    ).astype("Int64")
    game_possessions["period"] = pd.to_numeric(
        game_possessions["period"], errors="coerce"
    ).astype("Int64")
    game_possessions["possession_end_elapsed"] = pd.to_numeric(
        game_possessions["possession_end_elapsed"], errors="coerce"
    )
    game_possessions["points_scored"] = pd.to_numeric(
        game_possessions["points_scored"], errors="coerce"
    ).fillna(0).astype(int)
    game_possessions["event_count"] = pd.to_numeric(
        game_possessions["event_count"], errors="coerce"
    ).fillna(0).astype(int)
    game_possessions["_source_row"] = np.arange(len(game_possessions))
    game_possessions["possession_start_elapsed_raw"] = game_possessions["possession_start_elapsed"]
    game_possessions["possession_start_elapsed"] = game_possessions["possession_end_elapsed"].shift(
        fill_value=0
    )
    same_team = game_possessions["offense_team_id"].eq(
        game_possessions["offense_team_id"].shift()
    )
    same_period = game_possessions["period"].eq(game_possessions["period"].shift())
    same_end_time = game_possessions["possession_end_elapsed"].eq(
        game_possessions["possession_end_elapsed"].shift()
    )
    game_possessions["is_clock_stopped_continuation"] = (
        same_team & same_period & same_end_time & game_possessions["points_scored"].eq(1)
    )

    raptors_segment_mask = game_possessions["offense_team_id"] == RAPTORS_TEAM_ID
    raptors_possession_segments = game_possessions.loc[raptors_segment_mask].copy()
    raptors_possession_segments["SOURCE_POSSESSION_NUMBER"] = np.arange(
        1, len(raptors_possession_segments) + 1
    )
    raptors_possession_segments["RAPTORS_POSSESSION_NUMBER"] = (
        ~raptors_possession_segments["is_clock_stopped_continuation"]
    ).cumsum().astype(int)

    # Merge confirmed one-point continuations, as in the movement CSV producer.
    raptors_possessions = (
        raptors_possession_segments.groupby("RAPTORS_POSSESSION_NUMBER", as_index=False)
        .agg(
            period=("period", "first"),
            possession_start_elapsed=("possession_start_elapsed", "first"),
            possession_end_elapsed=("possession_end_elapsed", "last"),
            offense_team_id=("offense_team_id", "first"),
            points_scored=("points_scored", "sum"),
            event_count=("event_count", "sum"),
            source_possession_numbers=(
                "SOURCE_POSSESSION_NUMBER",
                lambda values: tuple(int(value) for value in values),
            ),
            contains_clock_stopped_continuation=(
                "is_clock_stopped_continuation", "any"
            ),
        )
    )
    raptors_possessions["possession_duration"] = (
        raptors_possessions["possession_end_elapsed"]
        - raptors_possessions["possession_start_elapsed"]
    )


    exported_points = raptors_movement_data.groupby("possession_number")["points_scored"]
    if exported_points.nunique().gt(1).any():
        raise ValueError("Movement export has conflicting points for a corrected possession.")
    raptors_possessions["points_scored_uncorrected"] = raptors_possessions["points_scored"]
    raptors_possessions["points_scored"] = (
        raptors_possessions["RAPTORS_POSSESSION_NUMBER"]
        .map(exported_points.first())
        .fillna(raptors_possessions["points_scored"])
        .astype(int)
    )


    # Match opponent turnovers and Raptors defensive rebounds using the original clock conversion.
    defense_forward_records = []
    unmatched_defensive_records = []
    last_matched_source_row = -1
    for loss in sorted(possession_losses, key=lambda item: item["LOSS_EVENTNUM"]):
        loss_event = game.loc[game["EVENTNUM"] == loss["LOSS_EVENTNUM"]].iloc[0]
        event_elapsed = float(loss_event["PCTIMESTRING"]) / 60.0
        candidates = raptors_possession_segments.loc[
            (raptors_possession_segments["period"] == int(loss_event["PERIOD"]))
            & (raptors_possession_segments["_source_row"] > last_matched_source_row)
            & ~raptors_possession_segments["is_clock_stopped_continuation"]
        ]
        if candidates.empty:
            unmatched_defensive_records.append({
                **loss, "REASON": "No later Raptors segment in this period",
                "NEAREST_OFFSET_SECONDS": np.nan,
            })
            continue

        offsets = (candidates["possession_start_elapsed"] - event_elapsed).abs()
        if offsets.min() > max_link_offset_seconds:
            unmatched_defensive_records.append({
                **loss, "REASON": "Nearest segment exceeds matching tolerance",
                "NEAREST_OFFSET_SECONDS": float(offsets.min()),
            })
            continue
        matched_segment = candidates.loc[offsets.idxmin()]
        last_matched_source_row = int(matched_segment["_source_row"])
        defense_forward_records.append(
            {
                "LOSS_EVENTNUM": int(loss["LOSS_EVENTNUM"]),
                "PERIOD": int(loss_event["PERIOD"]),
                "CAUSE": loss["CAUSE"],
                "DEFENSIVE_START_TYPE": (
                    "opponent turnover"
                    if "turnover" in loss["CAUSE"]
                    else "defensive rebound"
                ),
                "RAPTORS_POSSESSION_NUMBER": int(
                    matched_segment["RAPTORS_POSSESSION_NUMBER"]
                ),
                "SOURCE_POSSESSION_NUMBER": int(
                    matched_segment["SOURCE_POSSESSION_NUMBER"]
                ),
                "POSSESSION_LINK_OFFSET_SECONDS": float(offsets.min()),
            }
        )

    defense_forward_possession_details = pd.DataFrame(defense_forward_records, columns=[
        "LOSS_EVENTNUM", "PERIOD", "CAUSE", "DEFENSIVE_START_TYPE",
        "RAPTORS_POSSESSION_NUMBER", "SOURCE_POSSESSION_NUMBER",
        "POSSESSION_LINK_OFFSET_SECONDS",
    ])
    unmatched_defensive_events = pd.DataFrame(unmatched_defensive_records, columns=[
        "start_row", "PERIOD", "PCTIMESTRING", "LOSS_EVENTNUM", "OPPONENT_EVENT",
        "RAPTORS_TAKEAWAY", "CAUSE", "REASON", "NEAREST_OFFSET_SECONDS",
    ]).drop(columns="start_row")
    if defense_forward_possession_details["RAPTORS_POSSESSION_NUMBER"].duplicated().any():
        raise ValueError("Multiple defensive starts were matched to one Raptors possession.")

    defense_forward_possession_numbers = sorted(
        defense_forward_possession_details["RAPTORS_POSSESSION_NUMBER"].unique()
    )
    raptors_defense_forward_possessions = raptors_possessions.loc[
        raptors_possessions["RAPTORS_POSSESSION_NUMBER"].isin(
            defense_forward_possession_numbers
        )
    ].copy()
    raptors_scoring_defense_forward_possessions = raptors_defense_forward_possessions.loc[
        raptors_defense_forward_possessions["points_scored"] > 0
    ].copy()
    points_scored_through_defense = int(
        raptors_defense_forward_possessions["points_scored"].sum()
    )
    defense_forward_possession_details = defense_forward_possession_details.merge(
        raptors_defense_forward_possessions[
            ["RAPTORS_POSSESSION_NUMBER", "points_scored"]
        ].rename(columns={"points_scored": "POINTS_SCORED"}),
        on="RAPTORS_POSSESSION_NUMBER",
        how="left",
    )

    raptors_movement_data["SOURCE_POSSESSION_NUMBER"] = pd.to_numeric(
        raptors_movement_data["SOURCE_POSSESSION_NUMBER"], errors="coerce"
    ).astype("Int64")
    source_to_corrected_possession = raptors_possession_segments[
        ["SOURCE_POSSESSION_NUMBER", "RAPTORS_POSSESSION_NUMBER"]
    ].drop_duplicates()
    raptors_movement_data = raptors_movement_data.merge(
        source_to_corrected_possession,
        on="SOURCE_POSSESSION_NUMBER",
        how="left",
        validate="many_to_one",
    )
    # Zero-duration continuations (such as an and-one FT) have no movement samples.
    expected_movement_sources = set(
        raptors_possession_segments.loc[
            raptors_possession_segments["possession_end_elapsed"]
            > raptors_possession_segments["possession_start_elapsed"],
            "SOURCE_POSSESSION_NUMBER",
        ]
    )
    missing_movement_sources = expected_movement_sources - set(
        raptors_movement_data["SOURCE_POSSESSION_NUMBER"]
    )
    if raptors_movement_data["RAPTORS_POSSESSION_NUMBER"].isna().any():
        raise ValueError("Movement data contains unknown Raptors source segments.")
    if not raptors_movement_data["RAPTORS_POSSESSION_NUMBER"].eq(
        pd.to_numeric(raptors_movement_data["possession_number"])
    ).all():
        raise ValueError("Reconstructed possession IDs disagree with the movement export.")
    movement_coverage = raptors_possession_segments[[
        "SOURCE_POSSESSION_NUMBER", "RAPTORS_POSSESSION_NUMBER", "period",
        "possession_start_elapsed", "possession_end_elapsed",
        "is_clock_stopped_continuation",
    ]].copy()
    movement_coverage["movement_rows"] = (
        movement_coverage["SOURCE_POSSESSION_NUMBER"]
        .map(raptors_movement_data.groupby("SOURCE_POSSESSION_NUMBER").size())
        .fillna(0).astype(int)
    )
    movement_coverage["missing_positive_duration_movement"] = (
        movement_coverage["SOURCE_POSSESSION_NUMBER"].isin(missing_movement_sources)
    )

    raptors_defense_forward_movement_data = raptors_movement_data.loc[
        raptors_movement_data["RAPTORS_POSSESSION_NUMBER"].isin(
            defense_forward_possession_numbers
        )
    ].copy()
    raptors_scoring_defense_forward_movement_data = raptors_movement_data.loc[
        raptors_movement_data["RAPTORS_POSSESSION_NUMBER"].isin(
            raptors_scoring_defense_forward_possessions["RAPTORS_POSSESSION_NUMBER"]
        )
    ].copy()

    # Aliases using the project's broader meaning of a turnover (turnover or defensive rebound).
    raptors_possessions_after_turnovers = raptors_defense_forward_possessions
    raptors_movement_after_turnovers = raptors_defense_forward_movement_data
    raptors_clock_stopped_continuations = raptors_possession_segments.loc[
        raptors_possession_segments["is_clock_stopped_continuation"]
    ].copy()

    raptors_possession_summary = pd.Series(
        {
            "Raw Raptors possession segments": len(raptors_possession_segments),
            "Clock-stopped continuation segments merged": len(raptors_clock_stopped_continuations),
            "Total corrected Raptors possessions": len(raptors_possessions),
            "Beginning after opponent turnovers": (
                defense_forward_possession_details["DEFENSIVE_START_TYPE"]
                .eq("opponent turnover")
                .sum()
            ),
            "Beginning after defensive rebounds": (
                defense_forward_possession_details["DEFENSIVE_START_TYPE"]
                .eq("defensive rebound")
                .sum()
            ),
            "Total defense-forward possessions": len(raptors_defense_forward_possessions),
            "Defense-forward possessions where Toronto scored": len(raptors_scoring_defense_forward_possessions),
            "Points scored through defense-forward possessions": points_scored_through_defense,
        },
        name="COUNT",
    )

    return {
        "game_id": TRACKING_GAME_ID,
        "is_home": is_home,
        "converted_opponent_possession_losses": converted_opponent_possession_losses,
        "cause_summary": cause_summary,
        "raptors_possession_segments": raptors_possession_segments,
        "raptors_possessions": raptors_possessions,
        "defense_forward_possession_details": defense_forward_possession_details,
        "defense_forward_possession_numbers": defense_forward_possession_numbers,
        "raptors_defense_forward_possessions": raptors_defense_forward_possessions,
        "raptors_scoring_defense_forward_possessions": raptors_scoring_defense_forward_possessions,
        "points_scored_through_defense": points_scored_through_defense,
        "raptors_movement_data": raptors_movement_data,
        "raptors_defense_forward_movement_data": raptors_defense_forward_movement_data,
        "raptors_scoring_defense_forward_movement_data": raptors_scoring_defense_forward_movement_data,
        "raptors_possessions_after_turnovers": raptors_possessions_after_turnovers,
        "raptors_movement_after_turnovers": raptors_movement_after_turnovers,
        "raptors_clock_stopped_continuations": raptors_clock_stopped_continuations,
        "raptors_possession_summary": raptors_possession_summary,
        "unmatched_defensive_events": unmatched_defensive_events,
        "movement_coverage": movement_coverage,
    }
