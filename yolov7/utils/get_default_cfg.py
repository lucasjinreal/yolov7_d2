

def get_default_solver_configs(_C):
     # Set default optimizer
    _C.SOLVER.OPTIMIZER = "sgd"
    _C.SOLVER.LR_MULTIPLIER_OVERWRITE = []
    _C.SOLVER.WEIGHT_DECAY_EMBED = 0.0

    # Default world size in D2 is 0, which means scaling is not applied. For D2Go
    # auto scale is encouraged, setting it to 8
    assert _C.SOLVER.REFERENCE_WORLD_SIZE == 0
    _C.SOLVER.REFERENCE_WORLD_SIZE = 8
    # Besides scaling default D2 configs, also scale quantization configs
    _C.SOLVER.AUTO_SCALING_METHODS = [
        "default_scale_d2_configs",
        "default_scale_quantization_configs",
    ]
    return _C