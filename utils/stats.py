def get_model_size_str(model):
    nelem = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            if module.weight is not None:
                nelem += module.weight.numel()
        if hasattr(module, 'bias'):
            if module.bias is not None:
                nelem += module.bias.numel()
    size_str = '{:.2f} MB'.format(nelem * 4 * 1e-6)
    return size_str
