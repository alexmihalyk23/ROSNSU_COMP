import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from action_robot.action import Task
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np

class TaskActionServer(Node):
    def __init__(self):
        super().__init__('task_action_server')
        self._action_server = ActionServer(
            self,
            Task,
            'execute_task',
            self.execute_callback)
        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.green_detected = False
        self.moving = False
        self.goal_handle = None

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing task...')
        self.goal_handle = goal_handle

        task_type = goal_handle.request.task_type
        success = False

        if task_type == 1:
            success = self.go_to_green_light()
        else:
            self.get_logger().error('Unknown task type')

        goal_handle.succeed()
        result = Task.Result()
        result.success = success
        return result

    def go_to_green_light(self):
        self.get_logger().info('Going to green light...')
        self.green_detected = False
        self.moving = False

        while rclpy.ok():
            if self.green_detected and not self.moving:
                self.moving = True
                self.get_logger().info('Начинаем движение.')

            if self.moving:
                self.send_movement_command(0)  # Move forward
            else:
                self.send_stop_command()

            if self.green_detected:
                break

            self.send_feedback(0.5)  # Send feedback

        self.send_feedback(1.0)  # Send final feedback
        return True

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_image(cv_image)

    def process_image(self, cv_image):
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Определение диапазонов цветов для зеленого, белых и желтых линий
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        lower_white = np.array([0, 0, 240])
        upper_white = np.array([180, 30, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Создание масок для зеленого, белых и желтых линий
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Проверка на наличие зеленого цвета
        if cv2.countNonZero(mask_green) > 0:
            self.green_detected = True
            self.get_logger().info('Зеленый цвет найден!')
        else:
            self.green_detected = False
            self.get_logger().info('Зеленый цвет не найден.')

        # Объединение масок белых и желтых линий
        mask = cv2.bitwise_or(mask_white, mask_yellow)

        height, width, _ = cv_image.shape
        bottom_third = height - height // 4

        # Определяем точки для перспективного преобразования
        src_points = np.float32([[0, bottom_third], [width, bottom_third], [0, height], [width, height]])
        dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        # Выполняем перспективное преобразование
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(cv_image, M, (width, height))

        # Применяем маски к преобразованному изображению
        warped_hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
        warped_mask_white = cv2.inRange(warped_hsv_image, lower_white, upper_white)
        warped_mask_yellow = cv2.inRange(warped_hsv_image, lower_yellow, upper_yellow)

        # Объединяем маски
        warped_mask = cv2.bitwise_or(warped_mask_white, warped_mask_yellow)

        # Поиск контуров на маске
        contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow("image", cv_image)
        cv2.waitKey(1)

    def send_movement_command(self, error):
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = error / 100.0  # Пропорциональное управление поворотом
        self.publisher.publish(twist)
        self.get_logger().info(f'Отправлена команда на движение. Error: {error / 100.0}')

    def send_stop_command(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        self.get_logger().info('Отправлена команда на остановку.')

    def send_feedback(self, progress):
        if self.goal_handle is not None:
            feedback_msg = Task.Feedback()
            feedback_msg.progress = progress
            self.goal_handle.publish_feedback(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    task_action_server = TaskActionServer()
    rclpy.spin(task_action_server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
