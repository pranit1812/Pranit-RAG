import time
import json
from pathlib import Path

from config import get_config
from extractors.extraction_router import ExtractionRouter


def run_test(pdf_path: Path) -> dict:
    cfg = get_config()
    router = ExtractionRouter(cfg)
    results = {
        "file": str(pdf_path),
        "exists": pdf_path.exists(),
        "extractors": list(router.extractors.keys()),
        "suitable": [],
        "docling": {},
        "unstructured": {},
    }
    try:
        results["suitable"] = router._get_suitable_extractors(pdf_path)
    except Exception as e:
        results["suitable_error"] = str(e)

    if "docling" in router.extractors:
        ext = router.extractors["docling"]
        try:
            t0 = time.time()
            page = ext.parse_page(pdf_path, 0)
            t1 = time.time()
            tables = [b for b in page["blocks"] if b.get("type") == "table"]
            results["docling"] = {
                "ok": True,
                "elapsed_sec": round(t1 - t0, 2),
                "blocks": len(page["blocks"]),
                "tables": len(tables),
            }
        except Exception as e:
            results["docling"] = {"ok": False, "error": str(e)}
    else:
        results["docling"] = {"ok": False, "error": "not_available"}

    if "unstructured_hi_res" in router.extractors:
        ext = router.extractors["unstructured_hi_res"]
        try:
            t0 = time.time()
            page = ext.parse_page(pdf_path, 0)
            t1 = time.time()
            tables = [b for b in page["blocks"] if b.get("type") == "table"]
            results["unstructured"] = {
                "ok": True,
                "elapsed_sec": round(t1 - t0, 2),
                "blocks": len(page["blocks"]),
                "tables": len(tables),
            }
        except Exception as e:
            results["unstructured"] = {"ok": False, "error": str(e)}
    else:
        results["unstructured"] = {"ok": False, "error": "not_available"}

    return results


if __name__ == "__main__":
    sample = Path("storage/test2/raw/5853 Bway Drawings-100__DD.pdf")
    out = run_test(sample)
    print(json.dumps(out, indent=2))
