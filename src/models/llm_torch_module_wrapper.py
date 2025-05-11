from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder


#Defining how we want to sample the future trajectory:
#8 poses over 8 seconds with an interval of 1 second between each pose
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

# Custom wrapper class extending NuPlan's TorchModuleWrapper for use with LLM-based planners
class LLMTorchModuleWrapper(TorchModuleWrapper):
    def __init__(self, feature_builder):
        """
        Initialize the LLM Torch Module Wrapper with a specific feature builder.
        This sets up both the feature and target builders and defines how future trajectories should be sampled.
        """
        super().__init__(
            feature_builders=[feature_builder],  # Wrap the input feature builder in a list
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],  # Set up the trajectory target builder
            future_trajectory_sampling=trajectory_sampling,  # Define sampling strategy for future trajectory
        )
