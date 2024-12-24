import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from action_robot.action import Task

class TaskActionClient(Node):
    def __init__(self):
        super().__init__('task_action_client')
        self._action_client = ActionClient(self, Task, 'execute_task')

    def send_goal(self, task_type):
        goal_msg = Task.Goal()
        goal_msg.task_type = task_type

        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.progress}')

def main(args=None):
    rclpy.init(args=args)
    task_action_client = TaskActionClient()
    future = task_action_client.send_goal(1)  # Change task type here
    rclpy.spin_until_future_complete(task_action_client, future)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
