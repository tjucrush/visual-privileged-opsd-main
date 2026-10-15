# Public benchmark suite

The suite evaluates GUI point localization across six public data sources.

| Name in config | Official/public source | Adapter and evaluation scope |
|---|---|---|
| `screenspot` | [ScreenSpot](https://huggingface.co/datasets/bevaya/ScreenSpot) | HF `test` split, normalized xyxy |
| `screenspot-v2` | [OS-Copilot/ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) | Desktop/mobile/web JSON, **pixel xywh** |
| `screenspot-pro` | [likaixin/ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) | All application JSON, pixel xyxy |
| `groundui-1k` | [agent-studio/GroundUI-1K](https://huggingface.co/datasets/agent-studio/GroundUI-1K) | HF split is named `train` upstream; use as evaluation only, pixel xyxy |
| `osworld-g-bbox` | [xlang-ai/OSWorld-G](https://github.com/xlang-ai/OSWorld-G) | Only `box_type=bbox`, pixel xywh. Current upstream file has 470 bbox items; 54 refusal and 40 polygon items are excluded. **Not the full OSWorld-G score.** |
| `ui-vision-element` | [ServiceNow/ui-vision](https://huggingface.co/datasets/ServiceNow/ui-vision), [official code](https://github.com/uivision/UI-Vision) | Basic/functional/spatial element JSON, pixel xyxy; excludes layout/action tasks. Unified pooled element accuracy, not the official three-category macro average. |

Sources and example schemas were inspected on 2026-10-15. Pin a dataset revision/commit for experiments and record downloaded file hashes. Upstream schemas may change: adapters validate bounds and fail on unknown structures instead of guessing coordinate units. Canonical `source_annotation` preserves the source file for subgroup analysis.

## Export the two parquet-backed sources

```bash
pip install -e '.[data]'
python -m vpopsd.benchmarks --benchmark screenspot --output-dir data/benchmarks/screenspot
python -m vpopsd.benchmarks --benchmark groundui-1k --output-dir data/benchmarks/groundui-1k
```

Pass `--revision COMMIT_SHA` to pin the Hugging Face revision. The resolved dataset fingerprint is also saved. GroundUI's `train` split name does **not** mean it should enter the 7K training set.

## ScreenSpot-v2

Download `screenspotv2_image.zip` and the three JSON files from the linked upstream dataset; extract screenshots yourself into the image root, retaining their original filenames.

```bash
python -m vpopsd.benchmarks --benchmark screenspot-v2 --annotations data/raw/ssv2/screenspot_desktop_v2.json data/raw/ssv2/screenspot_mobile_v2.json data/raw/ssv2/screenspot_web_v2.json --image-root data/raw/ssv2/images --output-dir data/benchmarks/screenspot-v2
```

## ScreenSpot-Pro

Download the upstream `annotations/` and `images/` directories, extracting any upstream image archives. Pass every annotation JSON file. Shell glob expansion differs across platforms; use Python to expand explicitly:

```python
from pathlib import Path
from vpopsd.benchmarks import convert_annotations
from vpopsd.data import write_jsonl

rows = convert_annotations(
    sorted(Path("data/raw/sspro/annotations").glob("*.json")), "data/raw/sspro/images", "screenspot-pro"
)
write_jsonl("data/benchmarks/screenspot-pro/test.jsonl", rows)
```

## OSWorld-G bbox subset

Clone/download the official benchmark and obtain its screenshot files. Use the **original** `OSWorld-G.json`, not the refined-instruction variant, unless the variant is clearly renamed in your report:

```bash
python -m vpopsd.benchmarks --benchmark osworld-g-bbox --annotations data/raw/osworld/OSWorld-G.json --image-root data/raw/osworld/images --output-dir data/benchmarks/osworld-g-bbox
```

The adapter deliberately skips polygons and refusal examples because a point-in-rectangle metric would misrepresent those tasks. For a full official score, use the upstream task-specific evaluators instead.

## UI-Vision element grounding

Download/extract upstream screenshots and all three `annotations/element_grounding/` files. The image root should contain the `element_grounding/` screenshot directory.

```bash
python -m vpopsd.benchmarks --benchmark ui-vision-element --annotations data/raw/uivision/annotations/element_grounding/element_grounding_basic.json data/raw/uivision/annotations/element_grounding/element_grounding_functional.json data/raw/uivision/annotations/element_grounding/element_grounding_spatial.json --image-root data/raw/uivision/images --output-dir data/benchmarks/ui-vision-element
```

## Protocol

Use the same original screenshots, instruction template, resolution budget, greedy decoding and max-new-Tokens for base and distilled students. A point is correct if it lies inside its target box, including boundaries. Invalid and missing outputs count as incorrect; duplicate or foreign IDs cause an error. Metrics are fractions in [0,1], not percentages. The suite requires every configured dataset to exist, then reports individual scores, equal dataset macro accuracy and sample-weighted micro accuracy.

ScreenSpot and ScreenSpot-v2 share substantial source material; do not describe this suite as six independent distributions. Use held-out screenshot exclusions when preparing training data. Exclusions match decoded pixels exactly; they do not detect near-duplicate screenshots, resized copies or pretraining contamination.

Prepare all six canonical manifests before running `python -m vpopsd.suite` with `configs/benchmarks.yaml`.
