import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from ..models.llava_service import LlavaService
from ..models.diffusion_service import DiffusionService
from ..storage.gallery_service import GalleryService
from ..utils.security import validate_safe_image_path, to_portable_path, sanitize_path_string

class StudioAgent:
    """Intelligent agent orchestrator for Multimodal Studio using LlamaIndex AgentWorkflow."""

    def __init__(
        self,
        llava_service: Optional[LlavaService] = None,
        diffusion_service: Optional[DiffusionService] = None,
        gallery_service: Optional[GalleryService] = None,
        llm_model: str = "qwen3:4b-studio",
        api_base: str = "http://127.0.0.1:11434/v1",
        studio_root: Optional[Path] = None
    ):
        try:
            from llama_index.llms.openai_like import OpenAILike
            from llama_index.core.agent.workflow import AgentWorkflow, ToolCall, ToolCallResult
            from llama_index.core.memory import ChatMemoryBuffer
        except ImportError as e:
            raise ImportError(f"The 'llama-index' library is required to run StudioAgent: {e}")

        self.studio_root = studio_root or Path(__file__).resolve().parent.parent.parent
        self.llava = llava_service or LlavaService()
        self.diffusion = diffusion_service or DiffusionService()
        self.gallery = gallery_service or GalleryService()
        self._AgentWorkflow = AgentWorkflow
        
        self.llm = OpenAILike(
            model=os.getenv("OLLAMA_MODEL", llm_model),
            api_base=os.getenv("OLLAMA_BASE_URL", api_base),
            api_key="ollama",
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=4096,
            max_tokens=1024,
            timeout=180.0,
            temperature=0.1
        )
        
        self.memory = ChatMemoryBuffer.from_defaults()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> Any:
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
                clean_err = sanitize_path_string(str(e), studio_root=root)
                rec_id = gallery.add_record(
                    task_type="caption",
                    prompt=style,
                    parameters={"style": style, "image": portable_p},
                    output_text="",
                    latency_sec=0.0,
                    status="failed",
                    error_message=clean_err
                )
                return json.dumps({"error": clean_err, "status": "failed", "record_id": rec_id})

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
                clean_err = sanitize_path_string(str(e), studio_root=root)
                rec_id = gallery.add_record(
                    task_type="vqa",
                    prompt=question,
                    parameters={"question": question, "image": portable_p},
                    output_text="",
                    latency_sec=0.0,
                    status="failed",
                    error_message=clean_err
                )
                return json.dumps({"error": clean_err, "status": "failed", "record_id": rec_id})

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
                clean_err = sanitize_path_string(str(e), studio_root=root)
                rec_id = gallery.add_record(
                    task_type="transformation",
                    prompt=prompt,
                    parameters={"strength": strength, "image": portable_p},
                    output_text="",
                    latency_sec=0.0,
                    status="failed",
                    error_message=clean_err
                )
                return json.dumps({"error": clean_err, "status": "failed", "record_id": rec_id})

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

        return self._AgentWorkflow.from_tools_or_functions(
            [caption_image, answer_visual_question, transform_image_style, inspect_studio_gallery],
            llm=self.llm,
            system_prompt=system_prompt,
            timeout=300.0,
            verbose=False
        )

    async def run_request(self, user_prompt: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """Await completion and collect events concurrently so failed streams cannot hang UI."""
        import time
        from contextlib import suppress
        from llama_index.core.agent.workflow import ToolCall, ToolCallResult
        started = time.monotonic()
        handler = self.workflow.run(user_msg=user_prompt, memory=self.memory)
        traces = []
        completed_tools = []

        async def collect_events():
            async for event in handler.stream_events():
                if isinstance(event, ToolCall):
                    kwargs = {
                        key: sanitize_path_string(value, studio_root=self.studio_root)
                        if isinstance(value, str) else value
                        for key, value in event.tool_kwargs.items()
                    }
                    traces.append({"event": "ToolCall", "tool": event.tool_name, "kwargs": kwargs})
                elif isinstance(event, ToolCallResult):
                    output = sanitize_path_string(str(event.tool_output), studio_root=self.studio_root)
                    traces.append({"event": "ToolCallResult", "tool": event.tool_name, "output": output[:300]})
                    completed_tools.append({"tool": event.tool_name, "output": output})

        collector = asyncio.create_task(collect_events())
        try:
            response = await asyncio.wait_for(handler, timeout=300.0)
            # Some workflow versions leave their event iterator open after completion.
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(collector, timeout=5.0)
            text = str(response).strip()
            if not text and completed_tools:
                # Tool execution may succeed while a thinking model exhausts its
                # final-answer budget. Retry without tool schemas or prior memory.
                from llama_index.core.llms import ChatMessage
                summary = await asyncio.wait_for(self.llm.achat([
                    ChatMessage(role="system", content="Summarize these completed tool results concisely. Do not invent results."),
                    ChatMessage(role="user", content=json.dumps(completed_tools, ensure_ascii=False)),
                ]), timeout=180.0)
                text = (summary.message.content or "").strip()
                traces.append({"event": "FinalizationRetry", "tool": "llm", "output": text[:300]})
            if not text:
                raise RuntimeError("Agent returned no final answer; completed tool records remain saved in the gallery.")
            return text, traces, round(time.monotonic() - started, 3)
        finally:
            if not collector.done():
                collector.cancel()
            with suppress(asyncio.CancelledError):
                await collector
