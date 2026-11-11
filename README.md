# Multi-Modal Video Analysis

A proof-of-concept pipeline for **forensic analysis of road-traffic video** (dashcam and CCTV) using a multimodal LLM, **Gemini 2.5 Pro** called through **LiteLLM**. The model gets the full video plus selected still frames. A small **collision knowledge graph** adds domain knowledge to the prompt, and the model returns **structured, Pydantic-validated JSON**: the scene environment, every detected object, timestamped events, and a detailed breakdown for each collision or near-miss.

The project grew out of a series of Jupyter notebooks. Their logic has since been refactored into testable modules under `video_analysis_poc/video_llm/`.

---

## Why this exists

General-purpose video LLMs make predictable mistakes on collision footage. These are the cases the pipeline is built to catch:

- A vehicle stopped on a roundabout gets described as *"inexplicably stationary"*, when the stop was **caused** by an earlier sideswipe.
- A vehicle that **arrives after** the incident gets blamed for it.
- A lateral HGV swept-path scrape gets classified as a rear-end impact.
- The striking vehicle is missed because it **left the scene**.

The pipeline tackles these with four techniques:

1. **Camera-aware prompting.** Optical flow classifies the camera as static CCTV, forward dashcam or handheld, so the model knows whether background motion is real.
2. **Knowledge-graph RAG.** Retrieved collision archetypes come with explicit *ERRORS TO AVOID*, plus the traffic regulations that apply.
3. **Chain-of-thought output.** The model fills in a `reasoning_trace` (perception → vehicle inventory → interactions → causal conclusion) before it writes the structured answer.
4. **Two-pass verification.** A second LLM call re-checks the first answer against the top pattern's common misidentifications.

---

## Pipeline overview

```mermaid
flowchart LR
    V[Video file(s)] --> CD[CameraProfileDetector<br/>optical flow]
    V --> KF[KeyFrameExtractor<br/>stills at timestamps]
    H[Vehicle types, road type,<br/>keyword hints] --> R[KGRetriever]
    KG[(CollisionKnowledgeGraph<br/>NetworkX)] --> R
    CD --> PC[PromptComposer]
    R --> PC
    PC --> P1[Pass 1<br/>KG-augmented analysis]
    KF --> P1
    V --> P1
    P1 --> P2[Pass 2<br/>verification]
    P2 --> O[VideoSceneAnalysisOutput<br/>Pydantic]
    P1 -. traces .-> LF[Langfuse]
    P2 -. traces .-> LF
```

### Analysis modes

`VideoLLM` (in `llm_wrapper.py`) offers three modes, each with an orchestration function in `llm_caller.py`:

| Mode | `VideoLLM` method | Orchestrator in `llm_caller.py` | Output |
|---|---|---|---|
| Crash-only detection | `async_analyse_crash_footage` | `execute_video_analysis_stage` | `VideoAnalysisOutput` |
| Full scene description | `async_analyse_scene_footage` (+ `_batch`) | `execute_video_scene_analysis_stage` | `VideoSceneAnalysisOutput` |
| **KG-augmented two-pass** | `async_analyse_scene_footage_kg` | `execute_video_scene_analysis_kg_stage` | `VideoSceneAnalysisOutput` with `reasoning_trace` |

The first two modes take their prompts from **Langfuse prompt management**. The KG mode builds its prompts in code with `PromptComposer`; to steer it for a specific incident, pass `user_query`.

---

## Repository structure

```
.
├── pyproject.toml / poetry.lock        # Poetry project definition
├── conftest.py                         # puts the repo root on sys.path for pytest
├── tests/                              # 45 unit tests (CV, KG, prompt composer)
└── video_analysis_poc/
    ├── *.ipynb                         # PoC / demo notebooks (see below)
    ├── *.json, *.graphml               # saved example outputs and exported KGs
    ├── connection/langfuse/            # Langfuse client, prompt collation, PII masking stub
    ├── keyvault/                       # Azure Key Vault secret client
    └── video_llm/
        ├── llm_wrapper.py              # VideoLLM: async LiteLLM wrapper + Langfuse tracing
        ├── llm_caller.py               # stage-level orchestration functions
        ├── cv_preprocessing/
        │   ├── camera_detection.py     # CameraProfileDetector (Farneback optical flow)
        │   └── keyframe_extraction.py  # KeyFrameExtractor (base64 JPEG stills)
        ├── knowledge_graph/
        │   ├── graph.py                # CollisionKnowledgeGraph (+ default seed data, GraphML I/O)
        │   ├── retriever.py            # KGRetriever (weighted pattern scoring, regulation lookup)
        │   ├── formatter.py            # RAG context and verification text blocks
        │   └── models.py               # CollisionPattern, TrafficRegulation, ScoredPattern, ...
        ├── prompt_composer/composer.py # PromptComposer: system / task / verification prompts
        ├── schema/video.py             # Pydantic I/O schemas
        ├── langfuse_schema/            # ObservationPayload for tracing
        └── use_case_config/            # MetadataTemplate and related config dataclasses
```

---

## Core components

### `cv_preprocessing`

- **`CameraProfileDetector`** runs Farneback dense optical flow over the first `sample_seconds` of the video (default 5 s, capped at 150 frames).
  - Stability is `1 / (1 + mean_flow)`. Above `0.85` the camera is **static** (CCTV).
  - Otherwise, a flow coefficient of variation above `1.5` means **handheld**.
  - Anything else is **ego_forward** (dashcam).
  - It returns a `CameraProfile`, whose `prompt_context` is injected into the system prompt. Sync (`detect`) and thread-offloaded async (`async_detect`) versions are both provided.
- **`KeyFrameExtractor`** pulls frames at given timestamps, downscales them to `max_dimension` (default 768 px) and encodes them as base64 JPEG (quality 85). These stills give the model sharper spatial detail at critical moments.

### `knowledge_graph`

`CollisionKnowledgeGraph` wraps a `networkx.DiGraph`. The built-in seed (`from_defaults()`) has **19 nodes and 30 edges**:

- **Vehicle types:** car, truck, van, bus, motorcycle and pedestrian, each with a `swept_path_risk`.
- **Road geometries:** roundabout, motorway, intersection and urban road.
- **5 collision patterns:**
  - `roundabout_sideswipe_hgv`
  - `rear_end_stationary`
  - `hit_and_run_scrape`
  - `roundabout_failure_yield`
  - `lane_change_sideswipe`

  Each pattern has visual signatures, a causal chain and a list of `common_misidentifications`.
- **4 UK regulations:** `uk_roundabout_priority`, `uk_lane_discipline`, `uk_hgv_swept_path` and `uk_hit_and_run`.
- **Edge relations:** `INVOLVES`, `OCCURS_ON`, `OFTEN_FOLLOWED_BY` and `APPLIES_TO`.

You can build the graph from a dict (`from_dict`), load it from GraphML (`from_graphml`), or export it with `to_graphml` for import into Neo4j or Cosmos DB Gremlin.

`KGRetriever.retrieve_patterns` scores each pattern with configurable `RetrievalWeights`:

| Signal | Default weight |
|---|---|
| Matching vehicle type (per type) | +2 |
| Matching road geometry | +3 |
| Matching keyword hint (per keyword) | +1 |

`retrieve_regulations` returns the regulations for the jurisdiction that either apply to the road type through an `APPLIES_TO` edge or are marked `general`.

### `prompt_composer`

`PromptComposer` builds four pieces:

- **System prompt:** the camera context, causal-analysis principles and the CoT structure.
- **Task prompt:** the RAG context, the JSON output schema, instructions, and an optional `user_query`.
- **Verification prompt:** the Pass 1 JSON plus a misidentification check for the top pattern.
- **Per-key-frame annotations.**

### `schema`

These are the Pydantic models for inputs and outputs:

- `CameraSource`, with `camera_type` set to `dashcam_front`, `dashcam_rear` or `cctv`.
- `VideoAgentTemplate`. The model defaults to `gemini/gemini-2.5-pro`, with a 100 MB size limit and mp4/avi/mov/mkv input.
- `VideoSceneAnalysisOutput`, which contains `EnvironmentContext`, `VideoObjectObservation`, `SceneEvent`, `WitnessedCrashDetail`, `CrashInvolvedParty` and `ReasoningTrace`.

### `llm_wrapper.VideoLLM`

- Validates video size and format, then base64-encodes the video off the event loop.
- Builds the multimodal payloads for several cameras concurrently with `asyncio.gather`.
- Bounds batch concurrency with an `asyncio.Semaphore` (3 requests in flight by default).
- Calls `litellm.acompletion` with `response_format={"type": "json_object"}` and logs each call as a Langfuse generation.
- Langfuse records prompt and response text only if `capture_input` / `capture_output` are enabled. Token usage, the model name and model parameters are always recorded.

---

## Installation

Requires **Python 3.11+**, [Poetry](https://python-poetry.org/), and the FFmpeg/OpenCV runtime libraries your platform needs.

```bash
git clone https://github.com/NavidCOMSC/Multi-Modal-VideoAnalysis.git
cd Multi-Modal-VideoAnalysis

# --no-root: pyproject.toml declares a package name (`video_data_analysis`)
# that doesn't match the source directory (`video_analysis_poc`)
poetry install --no-root --with video,notebook,dev
```

Dependency groups:

| Group | Contents |
|---|---|
| main | pydantic, litellm, langfuse, langchain, Azure SDKs, numpy/pandas |
| `video` | opencv-python-headless, PyAV, ffmpeg-python, imageio, scikit-image, matplotlib |
| `notebook` | JupyterLab, ipykernel, ipywidgets |
| `azure` | Azure Monitor / OpenTelemetry instrumentation |
| `dev` | pytest, pytest-asyncio, pytest-cov, mypy, ruff, black, pre-commit |

> **Note:** `networkx` isn't declared directly. It comes in transitively through `scikit-image` in the `video` group, so install that group (or add `networkx` explicitly).

### Import paths

The CV, knowledge-graph, prompt-composer and schema modules import cleanly with only the repo root on `sys.path`. The LLM path (`llm_wrapper`, `llm_caller`, `connection.langfuse.prompts`) also uses some top-level imports (`use_case_config.*`, `connection.*`). Until those are made package-relative, extend `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PWD/video_analysis_poc:$PWD/video_analysis_poc/video_llm"
```

---

## Configuration

Credentials are read from a `MetadataTemplate` (`use_case_config/metadata.py`), which normally holds secrets resolved from Azure Key Vault. If no template is supplied, `VideoLLM` falls back to environment variables:

| Variable | Used by | Purpose |
|---|---|---|
| `LITELLM_SECRET` | `VideoLLM` | LiteLLM API key (fallback) |
| `LITELLM_URL` | `VideoLLM` | LiteLLM proxy base URL (fallback) |
| `ENV` | Langfuse client / prompt collation | `dev` / `uat` / `prod` → Langfuse environment and prompt label |
| `environment` | `keyvault/_az.py` | Key Vault environment selector |

Two pieces of setup are still yours to do:

- Set the Key Vault URI in `video_analysis_poc/keyvault/_az.py`. It's a placeholder for now.
- Authentication uses `DefaultAzureCredential`.

Keep credentials in a `.env` file (already git-ignored) or in Key Vault, never in notebooks or source.

---

## Usage

### Offline: camera detection, KG retrieval and prompt composition (no LLM call)

```python
from video_analysis_poc.video_llm.cv_preprocessing import CameraProfileDetector, KeyFrameExtractor
from video_analysis_poc.video_llm.knowledge_graph import CollisionKnowledgeGraph, KGRetriever
from video_analysis_poc.video_llm.prompt_composer import PromptComposer

video = "data/roundabout_dashcam.mp4"

camera_profile = CameraProfileDetector(sample_seconds=5).detect(video)
key_frames = KeyFrameExtractor(max_dimension=768).extract(video, [10, 15, 18, 20])

kg = CollisionKnowledgeGraph.from_defaults()
retriever = KGRetriever(kg)
patterns = retriever.retrieve_patterns(
    detected_vehicle_types=["truck", "car"],
    road_type="roundabout",
    keyword_hints=["sideswipe", "scrape"],
    top_k=3,
)
regulations = retriever.retrieve_regulations(road_type="roundabout", jurisdiction="UK")

composer = PromptComposer(camera_profile, patterns, regulations)
system_prompt = composer.build_system_prompt()
task_prompt = composer.build_task_prompt(user_query="Focus on the truck and SUV at 15–20s.")
```

### End to end: KG-augmented two-pass analysis

```python
import asyncio
from video_analysis_poc.connection.langfuse.tracing_connection import create_langfuse_client
from video_analysis_poc.video_llm.llm_caller import execute_video_scene_analysis_kg_stage
from video_analysis_poc.video_llm.schema.video import CameraSource, VideoAgentTemplate

metadata = ...  # a populated MetadataTemplate (LiteLLM + Langfuse config)

async def main():
    output = await execute_video_scene_analysis_kg_stage(
        metadata_instance=metadata,
        uuid_tag="INC-12345",
        langfuse_client=create_langfuse_client(metadata),
        video_agent=VideoAgentTemplate(task_template_name="video_scene_analysis_kg"),
        camera_sources=[CameraSource(camera_type="dashcam_front",
                                     video_file_name="data/roundabout_dashcam.mp4")],
        detected_vehicle_types=["truck", "car", "van"],
        detected_road_type="roundabout",
        keyword_hints=["sideswipe"],
        key_frame_timestamps=[10, 15, 18, 20, 25],
        run_verification_pass=True,
        user_query="Focus on the interaction between the truck and the black SUV at 15–20s.",
    )
    print(output.reasoning_trace.causal_conclusion)
    for event in output.events:
        print(event.timestamp_display, event.event_type, event.description)

asyncio.run(main())
```

If no `knowledge_graph` is passed, the default embedded graph is built for you.

### Example output (abridged)

```json
{
  "reasoning_trace": {
    "global_perception": "...",
    "vehicle_inventory": "...",
    "interaction_analysis": "...",
    "causal_conclusion": "..."
  },
  "environment": { "weather": "...", "lighting": "...", "road_type": "roundabout", "...": "..." },
  "detected_objects": [ { "object_type": "truck", "first_seen_seconds": 12.0, "confidence": 0.9, "...": "..." } ],
  "events": [
    {
      "event_type": "collision",
      "timestamp_seconds": 18.0,
      "timestamp_display": "00:00:18",
      "witnessed_crash_detail": {
        "impact_type": "sideswipe",
        "probable_cause": "...",
        "fault_indicators": ["..."],
        "dashcam_vehicle_involved": false
      }
    }
  ],
  "summary": "..."
}
```

Full saved outputs are in `video_analysis_poc/scene_analysis_*.json`.

---

## Notebooks

The notebooks trace how the approach developed. Each one adds a capability:

| Notebook | What it shows |
|---|---|
| `video_crash_analysis_demo.ipynb` | Baseline: crash-event detection with Gemini 2.5 Pro via LiteLLM |
| `video_scene_analysis_demo.ipynb` | Full scene description: all objects, environment and events |
| `video_scene_analysis_demo_causality.ipynb` | Causal-reasoning prompts plus key-frame supplementation |
| `video_scene_analysis_demo_logic_rag.ipynb` | Knowledge-graph RAG, chain-of-thought reasoning and verification pass |
| `video_analysis_integration_demo.ipynb` | The refactored module stack run end to end on a new video |
| `video_llm/modules_caller.ipynb` | Lists the models available behind the LiteLLM proxy |

Update the video paths and credentials at the top of each notebook before you run it.

---

## Testing

```bash
poetry run pytest            # or: python -m pytest -q
```

The suite has **45 tests** covering camera detection, key-frame extraction, KG construction and GraphML round-trips, retrieval scoring, formatting, and prompt composition. The tests use synthetic video, so you don't need an LLM, credentials or sample footage. The async tests need `pytest-asyncio`, which is in the `dev` group.

---

## Roadmap / known limitations

- **Vehicle types and road geometry are inputs.** Today they come from manual annotation; the natural next step is an upstream detector such as YOLO.
- **In-process NetworkX graph.** `CollisionKnowledgeGraph` is designed so the backend can be swapped for Neo4j or Cosmos DB Gremlin in production.
- **The KG covers UK regulations only.** Add regulation nodes to support other jurisdictions.
- **The PII masker is a placeholder.** `connection/langfuse/masking.py` needs a real anonymiser before trace capture is switched on.
- **Packaging.** The package name in `pyproject.toml` needs aligning with the source directory, and some imports need making package-relative (see [Import paths](#import-paths)).
