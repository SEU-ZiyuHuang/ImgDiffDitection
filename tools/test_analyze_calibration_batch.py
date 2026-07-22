"""Black-box test for the calibration-batch analysis command-line tool."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("analyze_calibration_batch.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CalibrationBatchAnalysisCommandTest(unittest.TestCase):
    """Validate the public CLI against a small completed/failed batch."""

    def test_reports_batch_coverage_reference_selection_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_root = Path(temporary_directory) / "batch"
            case_output = batch_root / "case_a"
            artifact_paths = {
                name: case_output / "components" / "component-000" / name
                for name in (
                    "alignment.png",
                    "valid_mask.png",
                    "difference_mask.png",
                    "difference_heatmap.png",
                    "annotated.png",
                )
            }
            for path in artifact_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"evidence")

            observation_path = case_output / "calibration_observation.json"
            _write_json(observation_path, {
                "mode": "calibration",
                "processing_profile_version": "development-default-v1",
                "selected_reference": {"id": "additional_0", "path": "additional.jpg"},
                "reference_attempts": [
                    {
                        "reference_id": "primary",
                        "reference_path": "primary.jpg",
                        "alignment_diagnostic": "unreliable",
                        "alignment_metrics": {"feature_match_count": 3},
                        "selected": False,
                    },
                    {
                        "reference_id": "additional_0",
                        "reference_path": "additional.jpg",
                        "alignment_diagnostic": "usable",
                        "alignment_metrics": {
                            "feature_match_count": 20,
                            "inlier_count": 15,
                            "inlier_rate": 0.75,
                            "reprojection_error_pixels": 1.2,
                            "spatial_coverage": 0.2,
                            "valid_overlap_ratio": 0.9,
                            "diagnostic": "usable",
                        },
                        "selected": True,
                    },
                ],
                "components": [
                    {
                        "component_index": 0,
                        "category": "meter",
                        "alignment_metrics": {
                            "feature_match_count": 20,
                            "inlier_count": 15,
                            "inlier_rate": 0.75,
                            "reprojection_error_pixels": 1.2,
                            "spatial_coverage": 0.2,
                            "valid_overlap_ratio": 0.9,
                            "diagnostic": "usable",
                        },
                        "difference_candidate_count": 2,
                        "difference_regions": [
                            {
                                "x": 1,
                                "y": 2,
                                "width": 4,
                                "height": 5,
                                "confidence": 0.8,
                                "evidence_channels": ["lab_color"],
                            },
                            {
                                "x": 7,
                                "y": 8,
                                "width": 3,
                                "height": 2,
                                "confidence": 0.6,
                                "evidence_channels": ["gradient_magnitude"],
                            },
                        ],
                        "observation_detail": "observed",
                        "artifacts": {
                            "alignment_image": str(artifact_paths["alignment.png"]),
                            "valid_mask": str(artifact_paths["valid_mask.png"]),
                            "difference_mask": str(artifact_paths["difference_mask.png"]),
                            "difference_heatmap": str(artifact_paths["difference_heatmap.png"]),
                            "annotated_image": str(artifact_paths["annotated.png"]),
                        },
                    }
                ],
            })
            _write_json(batch_root / "batch_summary_shard_00_of_01.json", {
                "mode": "calibration",
                "input_root": "input",
                "output_root": str(batch_root),
                "case_count_requested": 2,
                "counts": {"completed": 1, "failed": 1},
                "cases": [
                    {
                        "case": "case_a",
                        "output_directory": str(case_output),
                        "status": "completed",
                        "reference_count": 2,
                        "component_count": 1,
                        "selected_reference_id": "additional_0",
                        "manifest": str(observation_path),
                    },
                    {
                        "case": "case_b",
                        "output_directory": str(batch_root / "case_b"),
                        "status": "failed",
                        "error": "cannot decode live image",
                    },
                ],
            })

            report_directory = batch_root / "analysis"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--batch-output", str(batch_root),
                    "--output-directory", str(report_directory),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            report = json.loads(
                (report_directory / "calibration_analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["batch"]["requested_cases"], 2)
            self.assertEqual(report["batch"]["completed_cases"], 1)
            self.assertEqual(report["batch"]["failed_cases"], 1)
            self.assertEqual(report["references"]["additional_selected_cases"], 1)
            self.assertEqual(report["components"]["observed_components"], 1)
            self.assertEqual(report["evidence"]["complete_components"], 1)


if __name__ == "__main__":
    unittest.main()
