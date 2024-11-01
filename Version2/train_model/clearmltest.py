from clearml import Task

# Initialize a ClearML task
task = Task.init(project_name='Test Project', task_name='Test Task')

# Log some scalar values
logger = task.get_logger()
for i in range(10):
    logger.report_scalar(title='Test Metric', series='Series A', value=i, iteration=i)

print("ClearML test completed.")
