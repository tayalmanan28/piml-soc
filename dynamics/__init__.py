import importlib

def load_class(class_name):
    """
    Dynamically load a class by its name.
    """
    try:
        # Dynamically construct the module path
        module_name = f".{class_name}"
        module = importlib.import_module(module_name, package=__name__)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot find class '{class_name}'") from e