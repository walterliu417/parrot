**List of UCI options**\
\
explore_factor - How much the engine explores moves during the PUCT search. Setting too high could cause the engine to waste time searching useless moves, while setting too low might make the engine miss moves.\
capture_bonus - How much the engine explores captures during PUCT.\
check_bonus - How much the engine explores checking moves during PUCT.\
explore_decay - A value above 100 forces the engine to conclude exploration before the search time limit and only focus on the best moves during each rollout.\
tablebase_dir - Path to 5-piece Syzygy tablebases.\
net_path - Path to the neural network, stored as an Open Neural Network Exchange (ONNX) file.\
gpu_enabled - Set to true if GPU acceleration is available, otherwise false.\
num_threads - Number of threads to use - default is 1. If the user wishes to use the maximum number of threads, set this option to 0.\
temperature - Randomly perturbs the evaluation up to the percentage set by this parameter.\
temp_moves - Number of full moves to apply temperature pertubation to.\
datagen - Set datagen mode.\
log - Toggles console logging.
\
See technical.md for explanations on the search algorithm.