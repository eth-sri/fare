from box import Box

target_evaluations = Box({
    "num_runs": 5,
    "model_config": [
        Box({"name": "rf", "friendly_name": "random_forest"}),
    ]
})
