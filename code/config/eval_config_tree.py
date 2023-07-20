from box import Box

target_evaluations = Box({
    "num_runs": 5,
    "model_config": [
        Box({"name": "tree", "friendly_name": "decision_tree", "__tree__max_leaf_nodes": 100}),
    ]
})
