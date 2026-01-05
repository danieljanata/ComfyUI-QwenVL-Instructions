import os, sys, importlib.util

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

# Only expose the "Instructions" node(s) from this addon.
# Upstream modules are still importable for shared code (QwenVLBase, etc.),
# but their nodes are intentionally not registered here.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

def _load_module(py_filename: str):
    file_path = os.path.join(current_dir, py_filename)
    module_name = os.path.splitext(py_filename)[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {py_filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Ensure base code is importable (does not register nodes)
_load_module("AILab_QwenVL.py")

# Register only our node(s)
m = _load_module("AILab_QwenVL_Instructions.py")
if hasattr(m, "NODE_CLASS_MAPPINGS"):
    NODE_CLASS_MAPPINGS.update(m.NODE_CLASS_MAPPINGS)
if hasattr(m, "NODE_DISPLAY_NAME_MAPPINGS"):
    NODE_DISPLAY_NAME_MAPPINGS.update(m.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
