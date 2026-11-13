import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"


def notebook_source(name: str) -> str:
    payload = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_image_preprocessing_contract():
    paths = sorted(INPUTS.glob("*.png"))
    assert len(paths) == 3
    for path in paths:
        with Image.open(path) as image:
            prepared = image.convert("RGB")
            assert prepared.mode == "RGB" and prepared.size == (512, 512)
            assert prepared.size[0] % 8 == prepared.size[1] % 8 == 0


def test_pipeline_configuration_is_explicit():
    source = notebook_source("01_img2img_pipeline.ipynb")
    assert 'MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"' in source
    assert "StableDiffusionImg2ImgPipeline.from_pretrained" in source
    assert "torch_dtype=torch.float16" in source
    assert "enable_model_cpu_offload" in source


def test_controlled_experiment_parameters():
    source = notebook_source("02_parameter_experiments.ipynb")
    assert "strength_values = [0.3, 0.6, 0.9]" in source
    assert "cfg_values = [5.0, 7.5, 10.0]" in source
    assert "step_values = [20, 30, 50]" in source
    assert source.count("manual_seed(42)") >= 5
    assert "negative_prompt=NEGATIVE_PROMPT" in source


def test_metadata_to_image_consistency():
    metadata_files = sorted((OUTPUTS / "metadata").glob("*.json"))
    assert metadata_files
    for path in metadata_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        candidates = list((OUTPUTS / "images").rglob(record["output_image"]))
        assert candidates, f"Missing image for {path.name}"
        image_path = candidates[0]
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == record["output_sha256"]
        with Image.open(image_path) as image:
            assert list(image.size) == record["resolution"]
        assert 0 < float(record["strength"]) <= 1
        assert int(record["num_inference_steps"]) > 0


def test_analysis_reads_registry_instead_of_inventing_latency():
    source = notebook_source("03_results_analysis.ipynb")
    assert "pd.read_csv(registry_path)" in source
    assert "generation_time_seconds" in source


@pytest.mark.integration
def test_real_pipeline_smoke_inference():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; diffusion inference not executed")
    pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    image = Image.open(INPUTS / "01_coastal_lighthouse.png").convert("RGB").resize((256, 256))
    result = pipe(prompt="oil painting of a lighthouse", image=image, strength=0.2,
                  guidance_scale=5.0, num_inference_steps=2,
                  generator=torch.Generator(device="cpu").manual_seed(7)).images[0]
    assert result.size == (256, 256)
