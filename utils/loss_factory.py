import torch.nn as nn

def get_loss_function(loss_name):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "hierarchical_cross_entropy":
        # Placeholder: implement or import your custom loss here
        raise NotImplementedError("Hierarchical cross entropy not implemented.")
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")