"""Resume the known pretraining optimizer layouts without guessing parameter joins."""


def parameter_names(model, optimizer):
    names = {id(p): name.removeprefix("module.") for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def restore_optimizer(model, optimizer, checkpoint):
    saved = checkpoint["optimizer_state_dict"]
    parameters = {name.removeprefix("module."): p for name, p in model.named_parameters()}
    current_names = parameter_names(model, optimizer)
    saved_names = checkpoint.get("optimizer_param_names")
    if saved_names is None:
        all_names = list(parameters)
        trainable = [n for n in all_names if parameters[n].requires_grad]
        candidates = [current_names, [all_names], [trainable],
                      [[n for n in trainable if parameters[n].ndim <= 1],
                       [n for n in trainable if parameters[n].ndim > 1]],
                      [[n for n in all_names if not getattr(parameters[n], "_no_weight_decay", False)],
                       [n for n in all_names if getattr(parameters[n], "_no_weight_decay", False)]]]
        matches = {}
        for groups in candidates:
            if [len(g) for g in groups] != [len(g["params"]) for g in saved["param_groups"]]:
                continue
            valid = True
            for names, group in zip(groups, saved["param_groups"]):
                for name, key in zip(names, group["params"]):
                    for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        tensor = saved["state"].get(key, {}).get(field)
                        if tensor is not None and tuple(tensor.shape) != tuple(parameters[name].shape):
                            valid = False
            if valid:
                matches[tuple(tuple(g) for g in groups)] = groups
        if len(matches) != 1:
            raise ValueError("Cannot uniquely identify legacy optimizer parameter ordering; "
                             "load backbone weights for a new run instead of resuming")
        saved_names = next(iter(matches.values()))
    if len(saved_names) != len(saved["param_groups"]) or any(
            len(names) != len(group["params"])
            for names, group in zip(saved_names, saved["param_groups"])):
        raise ValueError("Checkpoint optimizer parameter names do not match its groups")
    flattened = [n for group in saved_names for n in group]
    if len(flattened) != len(set(flattened)) or any(n not in parameters for n in flattened):
        raise ValueError("Checkpoint optimizer parameter names are duplicated or unknown")
    ids = [key for group in saved["param_groups"] for key in group["params"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Checkpoint optimizer parameter IDs are duplicated")
    for names, group in zip(saved_names, saved["param_groups"]):
        for name, key in zip(names, group["params"]):
            for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                tensor = saved["state"].get(key, {}).get(field)
                if tensor is not None and tuple(tensor.shape) != tuple(parameters[name].shape):
                    raise ValueError(f"Checkpoint optimizer state shape mismatch for {name!r}")
    if saved_names == current_names:
        optimizer.load_state_dict(saved)
        return
    old = {name: (key, group) for names, group in zip(saved_names, saved["param_groups"])
           for name, key in zip(names, group["params"])}
    restored = optimizer.state_dict()
    restored["state"] = {}
    for names, group in zip(current_names, restored["param_groups"]):
        source_groups = []
        for name, key in zip(names, group["params"]):
            if name not in old:
                raise ValueError(f"Checkpoint optimizer is missing parameter {name!r}")
            old_key, source = old[name]
            if old_key in saved["state"]:
                restored["state"][key] = saved["state"][old_key]
            source_groups.append(source)
        if not source_groups:
            continue
        source = source_groups[0]
        fields = set(source) - {"params", "param_names", "weight_decay"}
        if any(any(g.get(k) != source[k] for k in fields) for g in source_groups):
            raise ValueError("Cannot merge optimizer groups with different hyperparameters")
        group.update({k: v for k, v in source.items() if k not in {"params", "param_names"}})
        if any(getattr(parameters[n], "_preserve_mamba_init", False) for n in names):
            flags = {bool(getattr(parameters[n], "_no_weight_decay", False)) for n in names}
            if flags == {True}:
                group["weight_decay"] = 0.0
    optimizer.load_state_dict(restored)
