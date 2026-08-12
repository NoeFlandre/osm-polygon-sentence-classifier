import hashlib

from osm_polygon_sentence_classifier.ablation_reporting import render_public_documents
from osm_polygon_sentence_classifier.ablation_study import (
    ABLATION_STUDY_ID,
    render_study_documents,
    study_specification,
    study_specification_fingerprint,
)
from osm_polygon_sentence_classifier.tracking import TRACKIO_STATIC_SPACE_ID


def test_public_report_renderer_preserves_existing_document_bytes() -> None:
    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
    )
    state = {
        "study_id": ABLATION_STUDY_ID,
        "fingerprint": study_specification_fingerprint(specification),
        "specification": specification,
        "phase": "running",
        "runs": {
            "a01-head-128|seed-42": {
                "ablation_id": "a01-head-128",
                "seed": 42,
                "run_id": "c" * 20,
                "phase": "completed",
                "metrics": {"eval_f1": 0.8, "eval_macro_f1": 0.7},
            }
        },
    }
    rows = [
        {
            "ablation_id": "a00-baseline-head-256-lr3e-4",
            "seed": 42,
            "status": "pending",
            "run_id": None,
            "source_commit": None,
            "metrics": {},
            "model_path": None,
        },
        {
            "ablation_id": "a01-head-128",
            "seed": 42,
            "status": "completed",
            "run_id": "c" * 20,
            "source_commit": None,
            "metrics": {"eval_f1": 0.8, "eval_macro_f1": 0.7},
            "model_path": "studies/landuse-v1/a01-head-128/run-cccccccccccccccccccc/final/",
        },
        *(
            {
                "ablation_id": ablation_id,
                "seed": 42,
                "status": "pending",
                "run_id": None,
                "source_commit": None,
                "metrics": {},
                "model_path": None,
            }
            for ablation_id in (
                "a02-head-512",
                "a03-head-lr1e-4",
                "a04-head-lr1e-3",
                "a05-balanced-head",
                "a06-last2-256",
            )
        ),
    ]

    documents = render_public_documents(
        state,
        rows=rows,
        study_id=ABLATION_STUDY_ID,
        tracking_space_id=TRACKIO_STATIC_SPACE_ID,
    )
    assert render_study_documents(state) == documents

    assert {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for path, content in documents.items()
    } == {
        "README.md": "6e1cbc4d998c2e801bfcc820b94a05cab0acd6b010080f16d2e17449c39c4055",
        "studies/landuse-v1/README.md": "13d604bcb3b0c6880e410e72bc9548b85c9d10a5c5516be44ae4589d426cbeba",
        "studies/landuse-v1/results.json": "58dceaec2cbebc32b30d0fb18544b95b8128f820936bba451526f1611d216e07",
        "studies/landuse-v1/study.json": "c7eb9dfa6dab3d0295334e78739a801fe8201166f401fa634985718bb191fbd2",
    }
