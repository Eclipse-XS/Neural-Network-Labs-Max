import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import AgentWorkflow, ToolCall, ToolCallResult
from llama_index.core.memory import ChatMemoryBuffer

from ..models.llava_service import LlavaService
from ..models.diffusion_service import DiffusionService
from ..storage.gallery_service import GalleryService
from ..utils.security import validate_safe_image_path, to_portable_path

class StudioAgent:
    """Intelligent agent orchestrator for Multimodal Studio using LlamaIndex AgentWorkflow."""

    def __init__(
        self,
        llava_service: Optional[LlavaService] = None,
        diffusion_service: Optional[DiffusionService] = None,
        gallery_service: Optional[GalleryService] = None,
        llm_model: str = "qwen3:4b-instruct",
        api_base: str = "http://127.0.0.1:11434/v1",
        studio_root: Optional[Path] = None
    ):
        self.studio_root = studio_root or Path(__file__).resolve().parent.parent.parent
        self.llava = llava_service or LlavaService()
        self.diffusion = diffusion_service or DiffusionService()
        self.gallery = gallery_service or GalleryService()
        
        self.llm = OpenAILike(
            model=llm_model,
            api_base=api_base,
            api_key="ollama",
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=4096,
            temperature=0.1
        )
        
        self.memory = ChatMemoryBuffer.from_defaults()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> AgentWorkflow:
        llava = self.llava
        diffusion = self.diffusion
        gallery = self.gallery
        root = self.studio_root

        def caption_image(image_path: str, style: str = "descriptive") -> str:
            """Generate a description or caption for an image file at image_path."""
            try:
                safe_p = validate_safe_image_path(image_path, studio_root=root)
                img = Image.open(safe_p)
                caption, latency = llava.generate_caption(img, style=style)
                portable_rel = to_portable_path(safe_p, studio_root=root)
                record_id = gallery.add_record(
                    task_type="caption",
                    prompt=style,
                    parameters={"style": style, "image": portable_rel},
                    output_text=caption,
                    input_image=img,
                    latency_sec=latency,
                    status="success"
                )
                return json.dumps({
                    "caption": caption,
                    "latency_sec": latency,
                    "record_id": record_id,
                    "image": portable_rel
                })
            except Exception as e:
                portable_p = to_portable_path(image_path, studio_root=root)
                rec_id = gallery.add_record(
                    task_type="caption",
                    prompt=style,
                    parameters={"style": style, "image": portable_p},
                    output_text="",
                    latency_sec=0.0,
                    status="failed",
                    error_message=str(e)
                )
                return json.dumps({"error": str(e), "status": "failed", "record_id": rec_id})

        def answer_visual_question(image_path: str, question: str) -> str:
            """Answer a question about the visual contents of the image at image_path."""
            try:
                safe_p = validate_safe_image_path(image_path, studio_root=root)
                img = Image.open(safe_p)
                answer, latency = llava.answer_question(img, question)
                portable_rel = to_portable_path(safe_p, studio_root=root)
                record_id = gallery.add_record(
                    task_type="vqa",
                    prompt=question,
                    parameters={"question": question, "image": portable_rel},
                    output_text=answer,
                    input_image=img,
                    latency_sec=latency,
                    status="success"
                )
                return json.dumps({
                    "answer": answer,
                    "latency_sec": latency,
                    "record_id": record_id,
                    "image": portable_rel
                })
            except Exception as e:
                portable_p = to_portable_path(image_path, studio_root=root)
                rec_id = gallery.add_record(
                    task_type="vqa",
                    prompt=question,
                    parameters={"question": question, "image": portable_p},
                    output_text="",
                    latency_sec=0.0,
                    status="failed",
                    error_message=str(e)
                )
                return json.dumps({"error": str(e), "status": "failed", "record_id": rec_id})

        def transform_image_style(image_path: str, prompt: str, strength: float = 0.6) -> str:
            """Apply creative image-to-image transformation using Stable Diffusion 1.5."""
            try:
                safe_p = validate_safe_image_path(image_path, studio_root=root)
                img = Image.open(safe_p)
                result_img, latency = diffusion.transform_image(img, prompt=prompt, strength=strength)
                portable_rel = to_portable_path(safe_p, studio_root=root)
                record_id = gallery.add_record(
                    task_type="transformation",
                    prompt=prompt,
                    parameters={"strength": strength, "image": portable_rel},
                    output_text=f"Transformed with prompt: '{prompt}' (strength={strength})",
                    input_image=img,
                    output_image=result_img,
                    latency_sec=latency,
                    status="success"
                )
                return json.dumps({
                    "status": "transformed",
                    "latency_sec": latency,
                    "record_id": record_id,
                    "output_image": f"{record_id}_output.png",
                    "source_image": portable_rel
                })
            except Exception as e:
                portable_p = to_portable_path(image_path, studio_root=root)
                rec_id = gallery.add_record(
                    task_type="transformation",
                    prompt=prompt,
                    parameters={"strength": strength, "image": portable_p},
                    output_text="",
                    latency_sec=0.0,
                    status="failed",
                    error_message=str(e)
                )
                return json.dumps({"error": str(e), "status": "failed", "record_id": rec_id})

        def inspect_studio_gallery(limit: int = 5) -> str:
            """Retrieve recent generation and analysis records from the studio gallery database."""
            records = gallery.get_all_records(limit=limit)
            summary = [
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"][:19],
                    "task_type": r["task_type"],
                    "prompt": r["prompt"],
                    "status": r.get("status", "success"),
                    "latency_sec": r["latency_sec"]
                }
                for r in records
            ]
            return json.dumps(summary, indent=2)

        system_prompt = (
            "You are the autonomous AI Director of the Multimodal Visual Content Studio at Lviv Polytechnic. "
            "You can analyze images with LLaVA (captioning and VQA), transform images using Stable Diffusion Img2Img, "
            "and inspect the studio history gallery. "
            "Autonomously select and sequence appropriate tools to satisfy the request. "
            "Do not execute destructive operations. Respect image path safety constraints."
        )

        return AgentWorkflow.from_tools_or_functions(
            [caption_image, answer_visual_question, transform_image_style, inspect_studio_gallery],
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=False
        )

    async def run_request(self, user_prompt: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """Execute user prompt through the agent workflow, capturing streaming events and traces."""
        import time
        t0 = time.time()
        handler = self.workflow.run(user_msg=user_prompt, memory=self.memory)
        traces = []
        
        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                # Clean any arguments from absolute paths for portable logging
                clean_kwargs = {}
                for k, v in event.tool_kwargs.items():
                    if isinstance(v, str) and (":" in v or "\\" in v):
                        clean_kwargs[k] = to_portable_path(v, studio_root=self.studio_root)
                    else:
                        clean_kwargs[k] = v
                traces.append({
                    "event": "ToolCall",
                    "tool": event.tool_name,
                    "kwargs": clean_kwargs
                })
            elif isinstance(event, ToolCallResult):
                traces.append({
                    "event": "ToolCallResult",
                    "tool": event.tool_name,
                    "output": str(event.tool_output)[:300]
                })
                
        response = await handler
        elapsed = time.time() - t0
        return str(response), traces, round(elapsed, 3)
