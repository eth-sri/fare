from box import Box

target_evaluations = Box({
    "num_runs": 5,
    "model_config": [
        Box({"name": "logreg", "friendly_name": "logistic_regression_normalized",
             "preprocess": "normalize", "__logreg__C": 1.0}),
    ]
})
