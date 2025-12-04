

if we might want to add Early Stopping (The Code). The StopTrainingOnNoModelImprovement callback must be passed to the callback_after_eval argument of the EvalCallback.

# 1. Define the Stopper
# "If the model doesn't beat the best score for 5 consecutive evaluations, stop."
stop_train_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=5, 
    min_evals=10, 
    verbose=1
)

# 2. Define the Evaluator (The Parent)
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./logs/',
    log_path='./logs/', 
    eval_freq=2000, 
    deterministic=True, 
    render=False,
    # <--- INJECT IT HERE
    callback_after_eval=stop_train_callback 
)

# 3. Train
model.learn(total_timesteps=100_000, callback=eval_callback)