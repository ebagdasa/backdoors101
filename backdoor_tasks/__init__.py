from backdoor_tasks.backdoor_task import BackdoorTask
from backdoor_tasks.pixel_pattern import PixelPattern


def get_backdoor_task(backdoor_task_params):
    backdoor_task_name = backdoor_task_params.get('attack_type', False)
    if backdoor_task_name == 'attack_type':
        return PixelPattern(backdoor_task_params)
    else:
        raise ValueError(f'Not supported task: {backdoor_task_name}')
