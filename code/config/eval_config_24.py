from box import Box

target_evaluations = Box({
    "num_runs": 1,
    "adversaries": True,
    "model_config": [
        ############### Pre-processing
        Box({
            "name": "nn", "friendly_name": "nn_1_layer_normalized",
            "__nn__hidden_layer_sizes": (50,), "__nn__max_iter": 1000,
            "__nn__n_iter_no_change": 10, "preprocess": "normalize",

            # uncomment for noop
            #'__nn__tol': 1e-3, '__nn__alpha': 0.1,
        }),
        Box({"name": "nn", "friendly_name": "nn_2_layer_normalized",
             "__nn__hidden_layer_sizes": (50, 50), "__nn__max_iter": 1000,
             "__nn__n_iter_no_change": 10, "preprocess": "normalize"
             }),

        Box({"name": "logreg", "friendly_name": "logistic_regression_normalized",
             "preprocess": "normalize", "__logreg__C": 1.0}),
        Box({"name": "rf", "friendly_name": "random_forest_normalized", "preprocess": "normalize"}),

        # ############################ Evaluation without pre-processing #######################
         Box({
             "name": "nn", "friendly_name": "nn_1_layer",
             "__nn__hidden_layer_sizes": (50,), "__nn__max_iter": 1000,
             "__nn__n_iter_no_change": 10,
         }),
          Box({"name": "nn", "friendly_name": "nn_2_layer",
               "__nn__hidden_layer_sizes": (50, 50), "__nn__max_iter": 1000,
               "__nn__n_iter_no_change": 10
               }),
         Box({"name": "logreg", "friendly_name": "logistic_regression",
              "__logreg__C": 1.0}),
        Box({"name": "rf", "friendly_name": "random_forest"}),

         ####################### Additional models
        Box({"name": "rf", "friendly_name": "random_forest_1000estimators", "__rf__n_estimators": 1000}),
        Box({"name": "tree", "friendly_name": "decision_tree", "__tree__max_leaf_nodes": 100}),
        Box({
            "name": "nn", "friendly_name": "nn_1_layer_normalized_WIDE",
            "__nn__hidden_layer_sizes": (200,), "__nn__max_iter": 1000,
            "__nn__n_iter_no_change": 10, "preprocess": "normalize",
        }),
        Box({"name": "nn", "friendly_name": "nn_2_layer_normalized_WIDE",
             "__nn__hidden_layer_sizes": (200, 100), "__nn__max_iter": 1000,
             "__nn__n_iter_no_change": 10, "preprocess": "normalize"
             }),
    ]
})
