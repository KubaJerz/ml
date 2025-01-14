import os
import importlib
import sys

def import_from_path(path, expected_class_name=None):
    try:
        dir_path, file_name = os.path.split(path)
        module_name = os.path.splitext(file_name)[0]
        class_name = expected_class_name or module_name

        sys.path.insert(0, dir_path)
        try:
            imported_module = importlib.import_module(module_name)
            imported_class = getattr(imported_module, class_name)
            return imported_class
        finally:
            sys.path.pop(0)

    except ImportError as e:
        raise ValueError(f"Failed to import module from '{path}': {e}")
    except AttributeError as e:
        raise ValueError(f"Failed to find class '{class_name}' in module: {e}")
    except Exception as e:
        raise ValueError(f"Error importing from path: {e}")