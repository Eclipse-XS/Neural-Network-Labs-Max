import os
import sys
import asyncio
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

# Add studio root directory to sys.path
STUDIO_ROOT = Path(__file__).resolve().parent
if str(STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_ROOT))

from src.utils.device import get_gpu_memory_info, clean_gpu_memory, ModelLifecycleManager
from src.utils.security import validate_safe_image_path, to_portable_path
from src.storage.gallery_service import GalleryService

st.set_page_config(
    page_title="Multimodal Visual Content Studio | Lviv Polytechnic",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistent gallery
@st.cache_resource
def get_gallery():
    return GalleryService()

gallery = get_gallery()

# Cached model services (lazy initialization)
@st.cache_resource
def get_llava_service():
    from src.models.llava_service import LlavaService
    return LlavaService()

@st.cache_resource
def get_diffusion_service():
    from src.models.diffusion_service import DiffusionService
    return DiffusionService()

# Sidebar: System telemetry & hardware status
with st.sidebar:
    st.title("🖥️ System Telemetry")
    mem_info = get_gpu_memory_info()
    st.metric("Device", mem_info.get("device", "CPU"))
    st.metric("VRAM Allocated", f"{mem_info.get('allocated_mb', 0)} MB")
    st.metric("VRAM Free", f"{mem_info.get('free_mb', 0)} MB")
    
    if st.button("🧹 Clear GPU Memory"):
        ModelLifecycleManager.get_instance().release_all()
        clean_gpu_memory()
        st.success("GPU memory cache cleared.")
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Academic Affiliation:**")
    st.markdown("Національний університет  \n«Львівська політехніка»")
    st.markdown("**Курс:** Технології та системи нейронних мереж")
    st.markdown("**Лабораторна робота №7 (Варіант 2)**")

# Main Header
st.title("🎨 Multimodal Visual Content Studio")
st.caption("Integrated Multimodal AI Studio: LLaVA Captioning & VQA • SD 1.5 Img2Img • LlamaIndex Autonomous Agent • Persistent Gallery")

tabs = st.tabs([
    "🖼️ Image Captioning",
    "❓ Visual QA",
    "✨ Image Transformation",
    "🤖 Agent Orchestration",
    "🏛️ Studio Gallery"
])

ASSETS_DIR = STUDIO_ROOT / "assets"
demo_images = sorted(list(ASSETS_DIR.glob("*.png")))
demo_options = {img.name: img for img in demo_images}

def get_selected_image(key_prefix: str) -> Image.Image:
    col1, col2 = st.columns([1, 1])
    with col1:
        use_demo = st.checkbox("Use Curated Demo Asset", value=True, key=f"{key_prefix}_use_demo")
        if use_demo and demo_options:
            selected_name = st.selectbox("Select Asset", list(demo_options.keys()), key=f"{key_prefix}_asset_select")
            img_path = demo_options[selected_name]
            image = Image.open(img_path)
        else:
            uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], key=f"{key_prefix}_upload")
            if uploaded:
                image = Image.open(uploaded)
            elif demo_options:
                image = Image.open(list(demo_options.values())[0])
            else:
                image = Image.new("RGB", (512, 512), color=(70, 70, 70))
    with col2:
        st.image(image, caption="Active Image Preview", width=280)
    return image

# -------------------------------------------------------------
# TAB 1: Image Captioning
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Multimodal Image Captioning (LLaVA-Interleave-Qwen-0.5B)")
    img_cap = get_selected_image("cap")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        cap_style = st.selectbox("Caption Style", ["concise", "descriptive", "detailed"], index=1)
    with col_c2:
        cap_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="cap_temp")
    with col_c3:
        cap_max_tokens = st.slider("Max New Tokens", 32, 256, 128, 16, key="cap_tokens")
        
    if st.button("Generate Caption", type="primary", key="btn_gen_cap"):
        with st.spinner("Running multimodal inference on LLaVA..."):
            try:
                llava = get_llava_service()
                caption, latency = llava.generate_caption(
                    img_cap, style=cap_style, temperature=cap_temp, max_new_tokens=cap_max_tokens
                )
                rec_id = gallery.add_record(
                    task_type="caption",
                    prompt=cap_style,
                    parameters={"style": cap_style, "temperature": cap_temp, "max_new_tokens": cap_max_tokens},
                    output_text=caption,
                    input_image=img_cap,
                    latency_sec=latency,
                    status="success"
                )
                st.success(f"Caption generated in {latency:.2f}s (Saved to Gallery: `{rec_id}`)")
                st.markdown(f"**Generated Caption:**\n\n> {caption}")
            except Exception as e:
                clean_gpu_memory()
                st.error(f"Inference error: {e}")

# -------------------------------------------------------------
# TAB 2: Visual Question Answering
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Visual Question Answering (VQA with LLaVA)")
    img_vqa = get_selected_image("vqa")
    
    question = st.text_input(
        "Enter your question about the image:",
        value="What is the primary object or activity depicted in this image?"
    )
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        vqa_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="vqa_temp")
    with col_v2:
        vqa_tokens = st.slider("Max Tokens", 32, 256, 128, 16, key="vqa_tokens")
        
    if st.button("Answer Question", type="primary", key="btn_vqa"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing visual evidence and generating answer..."):
                try:
                    llava = get_llava_service()
                    answer, latency = llava.answer_question(
                        img_vqa, question=question, temperature=vqa_temp, max_new_tokens=vqa_tokens
                    )
                    rec_id = gallery.add_record(
                        task_type="vqa",
                        prompt=question,
                        parameters={"question": question, "temperature": vqa_temp, "max_new_tokens": vqa_tokens},
                        output_text=answer,
                        input_image=img_vqa,
                        latency_sec=latency,
                        status="success"
                    )
                    st.success(f"Answer generated in {latency:.2f}s (Saved to Gallery: `{rec_id}`)")
                    st.markdown(f"**Answer:**\n\n{answer}")
                except Exception as e:
                    clean_gpu_memory()
                    st.error(f"Inference error: {e}")

# -------------------------------------------------------------
# TAB 3: Image Transformation (SD 1.5 Img2Img)
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Creative Image-to-Image Transformation (Stable Diffusion 1.5)")
    img_sd = get_selected_image("sd")
    
    sd_prompt = st.text_area(
        "Transformation Prompt:",
        value="An oil painting in the style of Vincent van Gogh, thick expressive impasto brushstrokes, vibrant swirling colors, masterpiece"
    )
    sd_neg = st.text_input("Negative Prompt:", value="blurry, low quality, distorted, deformed, artifacts, text")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        sd_strength = st.slider("Strength", 0.1, 0.9, 0.6, 0.05, key="sd_strength")
    with col_s2:
        sd_cfg = st.slider("CFG Scale", 3.0, 15.0, 7.5, 0.5, key="sd_cfg")
    with col_s3:
        sd_steps = st.slider("Inference Steps", 15, 50, 30, 5, key="sd_steps")
    with col_s4:
        sd_seed = st.number_input("Seed", value=42, step=1, key="sd_seed")
        
    if st.button("Transform Image", type="primary", key="btn_sd"):
        with st.spinner("Executing Stable Diffusion 1.5 Img2Img pipeline..."):
            try:
                diffusion = get_diffusion_service()
                result_img, latency = diffusion.transform_image(
                    img_sd,
                    prompt=sd_prompt,
                    negative_prompt=sd_neg,
                    strength=sd_strength,
                    guidance_scale=sd_cfg,
                    num_inference_steps=sd_steps,
                    seed=int(sd_seed)
                )
                rec_id = gallery.add_record(
                    task_type="transformation",
                    prompt=sd_prompt,
                    parameters={
                        "negative_prompt": sd_neg,
                        "strength": sd_strength,
                        "cfg": sd_cfg,
                        "steps": sd_steps,
                        "seed": sd_seed
                    },
                    output_text=f"Transformed with prompt: '{sd_prompt}' (strength={sd_strength})",
                    input_image=img_sd,
                    output_image=result_img,
                    latency_sec=latency,
                    status="success"
                )
                st.success(f"Transformation complete in {latency:.2f}s (Saved to Gallery: `{rec_id}`)")
                col_out1, col_out2 = st.columns(2)
                with col_out1:
                    st.image(img_sd, caption="Original Input Image", use_container_width=True)
                with col_out2:
                    st.image(result_img, caption="Transformed Output Image", use_container_width=True)
            except Exception as e:
                clean_gpu_memory()
                st.error(f"Transformation error: {e}")

# -------------------------------------------------------------
# TAB 4: Agent Orchestration (LlamaIndex Agent)
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Autonomous Workflow Orchestration (LlamaIndex Agent)")
    st.write(
        "The agent intelligently routes tasks between LLaVA captioning, VQA, Stable Diffusion Img2Img, "
        "and the Studio History Gallery based on user requests."
    )
    
    default_asset = "assets/01_coastal_lighthouse.png"
    agent_prompt = st.text_area(
        "Agent Instruction:",
        value=f"Inspect the image at '{default_asset}', provide a concise caption, and check the gallery records."
    )
    
    if st.button("Run Agent Workflow", type="primary", key="btn_agent"):
        with st.spinner("Autonomous agent planning and executing tool calls..."):
            try:
                from src.agent.studio_agent import StudioAgent
                agent = StudioAgent(
                    llava_service=get_llava_service(),
                    diffusion_service=get_diffusion_service(),
                    gallery_service=gallery,
                    studio_root=STUDIO_ROOT
                )
                response, traces, latency = asyncio.run(agent.run_request(agent_prompt))
                
                st.success(f"Agent finished in {latency:.2f}s")
                st.markdown(f"### Agent Final Response\n\n{response}")
                
                if traces:
                    with st.expander(f"🔍 Execution Trace Details ({len(traces)} events)"):
                        for tr in traces:
                            if tr["event"] == "ToolCall":
                                st.info(f"**🛠️ Tool Call:** `{tr['tool']}`\n\nArguments: `{tr['kwargs']}`")
                            elif tr["event"] == "ToolCallResult":
                                st.code(f"Output of {tr['tool']}:\n{tr['output']}", language="json")
            except Exception as e:
                clean_gpu_memory()
                st.error(f"Agent execution error: {e}")

# -------------------------------------------------------------
# TAB 5: Studio Gallery & Audit
# -------------------------------------------------------------
with tabs[4]:
    st.subheader("Persistent Studio History & Gallery Database")
    records = gallery.get_all_records(limit=30)
    
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        st.write(f"Total persisted records found: **{len(records)}** (Stored in `outputs/gallery/gallery.db`)")
    with col_g2:
        if st.button("🔄 Refresh Gallery"):
            st.rerun()
            
    if records:
        df_records = pd.DataFrame([
            {
                "ID": r["id"],
                "Timestamp": r["timestamp"][:19],
                "Task": r["task_type"],
                "Prompt / Subject": (r["prompt"] or "")[:50],
                "Latency (s)": round(r["latency_sec"] or 0.0, 2),
                "Status": r["status"]
            }
            for r in records
        ])
        st.dataframe(df_records, use_container_width=True)
        
        st.markdown("### Browsable Visual Artifacts")
        gallery_dir = STUDIO_ROOT / "outputs" / "gallery"
        for r in records[:6]:
            with st.container():
                st.markdown(f"**Record ID:** `{r['id']}` | **Task:** `{r['task_type']}` | **Latency:** {r['latency_sec']}s")
                cols = st.columns([1, 1, 2])
                with cols[0]:
                    if r.get("input_image_path"):
                        in_p = gallery_dir / r["input_image_path"]
                        if in_p.exists():
                            st.image(str(in_p), caption="Input Image", width=180)
                with cols[1]:
                    if r.get("output_image_path"):
                        out_p = gallery_dir / r["output_image_path"]
                        if out_p.exists():
                            st.image(str(out_p), caption="Output Image", width=180)
                with cols[2]:
                    if r.get("output_text"):
                        st.info(r["output_text"])
                st.divider()
    else:
        st.info("No studio records in database yet. Generate captions, VQA, or transformations to populate the gallery.")
