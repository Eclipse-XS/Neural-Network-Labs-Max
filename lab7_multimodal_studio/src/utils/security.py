import os
import re
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

def to_portable_path(image_path: str | Path, studio_root: Optional[Path] = None) -> str:
    """Convert an absolute or relative file path into a clean, portable project-relative path."""
    if studio_root is None:
        studio_root = Path(__file__).resolve().parent.parent.parent
    p = Path(image_path)
    try:
        if p.is_absolute():
            rel = p.resolve().relative_to(studio_root.resolve())
            return str(rel).replace("\\", "/")
        else:
            return str(image_path).replace("\\", "/")
    except ValueError:
        return Path(image_path).name

def sanitize_path_string(text: str, studio_root: Optional[Path] = None) -> str:
    """
    Sanitize text by replacing host-specific absolute paths with portable representations.
    Ensures no private Windows drive paths, user-home directories, or internal repository paths leak.
    """
    if not text:
        return text

    if studio_root is None:
        studio_root = Path(__file__).resolve().parent.parent.parent

    sanitized = text

    # Collect known sensitive paths
    targets = []

    # 1. Studio root and its variants
    res_root = str(studio_root.resolve())
    targets.append((res_root, "<studio_root>"))
    targets.append((studio_root.resolve().as_posix(), "<studio_root>"))
    targets.append((res_root.replace("\\", "\\\\"), "<studio_root>"))

    # 2. Repo root and variants
    try:
        repo_root = studio_root.parent.resolve()
        targets.append((str(repo_root), "<repo_root>"))
        targets.append((repo_root.as_posix(), "<repo_root>"))
        targets.append((str(repo_root).replace("\\", "\\\\"), "<repo_root>"))
    except Exception:
        pass

    # 3. User home directory
    try:
        home = Path.home().resolve()
        targets.append((str(home), "<user_home>"))
        targets.append((home.as_posix(), "<user_home>"))
        targets.append((str(home).replace("\\", "\\\\"), "<user_home>"))
    except Exception:
        pass

    # Sort targets by length descending to replace longest prefixes first
    targets.sort(key=lambda x: len(x[0]), reverse=True)
    for target, repl in targets:
        if target:
            sanitized = sanitized.replace(target, repl)

    # Regex cleanup for any residual Windows drive-letter user paths: e.g. C:\Users\... or C:\\Users\\...
    sanitized = re.sub(r'[A-Za-z]:\\\\Users\\\\[^\\]+\\\\', lambda m: '<user_home>\\\\', sanitized)
    sanitized = re.sub(r'[A-Za-z]:\\Users\\[^\\]+\\', lambda m: '<user_home>\\', sanitized)
    sanitized = re.sub(r'/Users/[^/]+/', lambda m: '<user_home>/', sanitized)
    sanitized = re.sub(r'/home/[^/]+/', lambda m: '<user_home>/', sanitized)

    return sanitized

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
        
    portable_ref = to_portable_path(image_path, studio_root=studio_root)

    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: '{portable_ref}'")
        
    if p.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension '{p.suffix}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}")
        
    if allowed_dirs is None:
        allowed_dirs = get_allowed_base_dirs(studio_root)
        
    # Check that resolved path is inside at least one allowed directory
    is_safe = any(
        p == d or d in p.parents
        for d in allowed_dirs
    )
    if not is_safe:
        raise SecurityValidationError(
            f"Access denied: '{portable_ref}' resolves outside allowed studio image directories."
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
        raise ValueError(f"Corrupted or invalid image file '{portable_ref}': {e}")
        
    return p
