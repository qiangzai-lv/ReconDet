from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class UVPointVisualizationHook(Hook):
    """Save raw-image UV overlays every N training iterations."""

    def __init__(self, interval=10, num_views=10):
        self.interval = interval
        self.num_views = num_views

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        iteration = runner.iter + 1
        model._uv_visualization_iter = (
            iteration if runner.rank == 0 and iteration % self.interval == 0
            else None)
        model._uv_visualization_done = False
        model._uv_visualization_num_views = self.num_views
        model._uv_visualization_dir = runner.work_dir
