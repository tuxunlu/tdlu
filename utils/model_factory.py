def get_model_class(model_name):
    if model_name == "MGModule":
        from models import MGModule
        return MGModule
    elif model_name == "MGModule_SingleHead":
        from models import MGModule_SingleHead
        return MGModule_SingleHead
    elif model_name == "MGModuleViT":
        from models import MGModuleViT
        return MGModuleViT
    else:
        raise ValueError(f"Unknown model name: {model_name}")