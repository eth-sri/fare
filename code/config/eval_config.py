from box import Box

target_evaluations = Box({
    "num_runs": 5,
    "model_config": [
        Box({
            "name": "nn", "friendly_name": "nn_1_layer_normalized",
            "__nn__hidden_layer_sizes": (50,), "__nn__max_iter": 1000,
            "__nn__n_iter_no_change": 10, "preprocess": "normalize",

            # uncomment for noop
            #'__nn__tol': 1e-3, '__nn__alpha': 0.1,
        })
    ]
})
