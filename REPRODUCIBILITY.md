# Reproducing Labs 1-7

## Supported environment

Windows 11 x64, PowerShell, Python **3.13**, and one isolated `.venv` per lab. `python` need not be on PATH: setup discovers the interpreter through `py -3.13`, or accepts `-Python <executable>`. Direct requirements and complete Windows/Python 3.13 transitive locks are committed for every lab. Do not combine the lab environments: LangChain, agents, notebooks and model stacks have different constraints.

```powershell
.\setup.ps1 -Lab 1
.\run.ps1 -Lab 1 -Smoke
.\run.ps1 -Lab 1
```

Repeat with the required lab number. Labs 1, 3-6 run their preserved notebooks in order in one private kernel; this also supplies state shared by consecutive Max notebooks. An executed copy, including partial output on failure, goes to the ignored `outputs/reproducibility-run.ipynb`. Lab 2 opens its CLI. Lab 7 opens Streamlit. Notebook experimental matrices can take substantially longer than the short functional tests.

## NVIDIA RTX 3050 4 GB

For Labs 4-5 and the Max studio, use `setup.ps1 -Lab <number> -Cuda`. This selects **torch 2.14.1+cu130** from `https://download.pytorch.org/whl/cu130`; other dependencies remain locked. It requires a driver compatible with CUDA 13.0. The wheel supplies its runtime; a separate CUDA toolkit is not required. The verification machine reports driver **610.88**. CPU environments use torch 2.14.1+cpu. NNTS Lab 7 deliberately embeds on CPU.

Run batch size 1 at 512x512. SD uses FP16 and CPU offload on CUDA. The Max studio unloads the other vision model before switching and serializes complete inference calls across Streamlit sessions. Its Ollama model runs on CPU. Do not run a second GPU workload concurrently. The 16 GB RAM verification machine also experienced paging when both LM Studio and Ollama were loaded alongside multiple model processes; heavy verification was serialized. CPU-only vision performance and different driver/Python combinations are not covered by the GPU evidence.

## Model and input integrity

`models.lock.json` records immutable Hugging Face revisions and upstream SHA-256 values for large model files. `scripts/fetch_models.py` downloads the selected revision and verifies every downloaded large file against that record. Normal model downloads are part of setup. Downloads require network access; no file from an earlier project directory is required. Existing standard Hugging Face caches may be reused after checksum verification.

```powershell
# Use the selected lab interpreter, for example:
.\lab7_multimodal_studio\.venv\Scripts\python.exe scripts/fetch_models.py llava
.\lab7_multimodal_studio\.venv\Scripts\python.exe scripts/fetch_models.py diffusion --fp16
.\lab7_multimodal_studio\.venv\Scripts\python.exe scripts/fetch_models.py minilm
```

The original LLaVA 0.5B and SD 1.5 checkpoints are used, not tiny substitute models. The model downloader verifies provenance, not semantic correctness or bit-identical GPU output. Seeds, versions and source revisions establish the reproducible configuration; GPU kernels can still differ between hardware/runtime combinations.

`inputs.sha256.json` records the valuable inputs already distributed in Git. Text checksums normalize CRLF to LF; binaries are checked byte-for-byte. Experimental notebooks, examples, gallery samples, datasets and benchmark outputs are preserved. Virtual environments, model caches, smoke images/databases and executed notebook artifacts are ignored.

## Service provisioning

Service identities/digests are recorded in `services.lock.json`. Never expose the unauthenticated local endpoints publicly.

Install and start Ollama on http://127.0.0.1:11434. Setup for Labs 2, 3, 6 and 7 pulls the public `qwen3:4b` weights, verifies the recorded source manifest digest, then reconstructs `qwen3:4b-studio` from `lab7_multimodal_studio/Modelfile`. That file sets CPU residency and a 4096-token context, retaining the upstream Jinja template required for tool calling. It has no dependency on the previously local `qwen3:4b-instruct` alias or a local GGUF path. Do not replace its template with a legacy Go template: the tested Ollama runner rejected tools with `tools param requires --jinja flag`.

`OLLAMA_MODEL` and `OLLAMA_BASE_URL` override model/endpoint defaults. Thinking models may spend part of their completion budget before producing a final response; the functional client smoke uses a sufficient budget instead of assuming a 16-token completion is enough. The Max CLI retains `--base-url` / `--model` support for a separate vLLM server. Native Windows reproduction uses Ollama; the WSL2/vLLM serving scenario was not executed in this verification.

Lab 7 standalone: `lab7_multimodal_studio/setup.ps1`, then `lab7_multimodal_studio/run.ps1`. Its UI listens on http://localhost:8502. The default setup uses CUDA; `-Cpu` selects CPU PyTorch. `-SkipModels` / `-SkipServices` are deliberate component-test options and do not establish complete provisioning. Gallery SQLite and image files are anchored to this checkout. Existing committed gallery data is historical evidence, not a required external database. Lab 1 datasets and all image/runbook/corpus inputs are already in Git and checked by the input manifest.

## Verification scope

`run.ps1 -Lab N -Smoke` runs real functionality and fails when a required service/model is unavailable. Lab 1 takes forward/backward/optimizer steps through both real model classes; Max also reads its distributed Enron data. Lab 2 calls the actual client implementation. Lab 3 builds/searches real Redis or Chroma vectors and calls the local LLM. Lab 4 generates a real 512x512 image using the original SD checkpoint; eight diffusion steps and a nonblack-output check test execution, not output quality. Lab 5 performs real visual inference. Lab 6 requires an actual agent tool call. Lab 7 tests real model/service workflows, SQLite/image persistence and Streamlit rendering.

These smoke tests do not retrain complete historical experiments or revalidate every experimental metric. Historical notebook outputs and timings remain labeled as historical. Passing imports or injected regression responses are not counted as live inference. See `VERIFICATION.md` for the exact executed record and limitations.

## Updating locks

After intentionally changing direct requirements, resolve each lab separately with `uv pip compile <lab>/requirements.txt --python-version 3.13 --python-platform windows --output-file <lab>/requirements.lock.txt`. Install in a clean lab environment, run the live smoke, then commit both requirement files. `uv` is a developer convenience; the PowerShell setup scripts use standard Python/pip. Do not regenerate model revisions/checksums implicitly on application startup.

## Studio UI and process persistence checks

```powershell
.\lab7_multimodal_studio\.venv\Scripts\python.exe scripts/smoke_studio_ui.py
.\lab7_multimodal_studio\.venv\Scripts\python.exe -m pytest lab7_multimodal_studio/tests -q
```

The UI check clicks real caption and Img2Img controls. Only the gallery constructor is redirected to temporary storage; inference, SQLite, PNG output and callbacks are real. The persistence test reopens the gallery in a separate Python process and checks that Windows releases its SQLite handles.
