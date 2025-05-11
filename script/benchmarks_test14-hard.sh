PLANNER="Align2Act"
BENCHMARK='test14-hard' # test14-random
CHALLENGES=$1 #closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_simulation.py \
    +simulation=$CHALLENGES \ 
    planner=Align2Act \
    worker.threads_per_node=8 \
    scenario_builder=nuplan_challenge \
    scenario_filter=$BENCHMARK \
    experiment_uid=$BENCHMARK/$PLANNER \
    verbose=true 
