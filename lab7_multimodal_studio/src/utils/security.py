import os
from pathlib import Path
from typing import List, Optional
from PIL import Image

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_DIMENSION = 2048

class SecurityValidationError(PermissionError):
    """Raised when an image path fails security validation."""
    pass

def get_allowed_base_dirs(studio_root: Optional[Path] = None) -> List[Path]:
    """Return the list of directories from which images may be safely loaded."""
    if studio_root is None:
        studio_root = Path(__file__).resolve().parent.parent.parent
    return [
        (studio_root / "assets").resolve(),
        (studio_root / "outputs" / "gallery").resolve(),
        (studio_root / "inputs").resolve()
    ]

def validate_safe_image_path(
    image_path: str | Path,
    studio_root: Optional[Path] = None,
    allowed_dirs: Optional[List[Path]] = None
) -> Path:
    """
    Validate that image_path:
    1. Points to an existing file.
    2. Has an allowed image file extension.
    3. Resolves strictly inside the permitted directories (preventing path traversal).
    4. Is a valid, uncorrupted image file within safe dimension limits.
    """
    if studio_root is None:
        studio_root = Path(__file__).resolve().parent.parent.parent
        
    p = Path(image_path)
    # If a relative path is given, resolve it relative to studio_root
    if not p.is_absolute():
        p = (studio_root / p).resolve()
    else:
        p = p.resolve()
        
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: '{image_path}'")
        
    if p.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension '{p.suffix}'. Allowed: {ALLOWED_IMAGE_EXTENSIONS}")
        
    if allowed_dirs is None:
        allowed_dirs = get_allowed_base_dirs(studio_root)
        
    # Check that resolved path is inside at least one allowed directory
    is_safe = any(
        p == d or d in p.parents
        for d in allowed_dirs
    )
    if not is_safe:
        raise SecurityValidationError(
            f"Access denied: '{image_path}' resolves outside allowed directories: "
            f"{[str(d) for d in allowed_dirs]}"
        )
        
    # Verify image integrity and dimensions
    try:
        with Image.open(p) as img:
            img.verify()
        # Re-open for dimension check since verify() closes/invalidates the handle
        with Image.open(p) as img:
            w, h = img.size
            if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                raise ValueError(f"Image dimensions ({w}x{h}) exceed maximum allowed ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})")
    except Exception as e:
        if isinstance(e, (ValueError, PermissionError)):
            raise
        raise ValueError(f"Corrupted or invalid image file '{image_path}': {e}")
        
    return p

def to_portable_path(image_path: str | Path, studio_root: Optional[Path] = None) -> str:
    """Convert an absolute file path into a clean, portable project-relative path."""
    if studio_root is None:
        studio_root = Path(__file__).resolve().parent.parent.parent
    p = Path(image_path).resolve()
    try:
        rel = p.relative_to(studio_root.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return Path(image_path).name
