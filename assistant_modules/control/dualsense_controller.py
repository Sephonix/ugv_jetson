# vibe coded using claude sonnet 4.5...

import pygame
import time
from collections import deque
from typing import Optional, Callable

import audio_ctrl

class DualSenseController:
    """
    DualSense controller interface for rover control

    Args:
        base_controller: Instance of base_ctrl.BaseController
        deadzone: Joystick deadzone threshold (default: 0.15)
        gimbal_speed: Gimbal movement speed (default: 50)
        max_wheel_speed: Maximum wheel speed (default: 0.2)
    """

    # DualSense axis mappings
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_LEFT_L2 = 2
    AXIS_RIGHT_X = 3
    AXIS_RIGHT_Y = 4
    AXIS_RIGHT_L2 = 5
    AXIS_DPAD_X = 6
    AXIS_DPAD_Y = 7

    # DualSense button mappings
    BUTTON_CROSS = 0
    BUTTON_CIRCLE = 1
    BUTTON_TRIANGLE = 2
    BUTTON_SQUARE = 3
    BUTTON_L1 = 4
    BUTTON_R1 = 5
    BUTTON_L2 = 6
    BUTTON_R2 = 7
    BUTTON_SHARE = 8  # aka the SELECT button
    BUTTON_OPTIONS = 9  # aka the START button
    BUTTON_MODE = 10  # aka the PS button
    BUTTON_L3 = 11  # aka the left stick press
    BUTTON_R3 = 12  # aka the right stick button

    def __init__(
        self, base_controller, deadzone=0.05, gimbal_speed=50, max_wheel_speed=0.2
    ):
        self.base = base_controller
        self.deadzone = deadzone
        self.gimbal_speed = gimbal_speed
        self.max_wheel_speed = max_wheel_speed

        # initialize pygame if it isnt already
        if not pygame.get_init():
            pygame.init()

        if not pygame.joystick.get_init():
            pygame.joystick.init()

        # controller instance
        self.joystick: Optional[pygame.joystick.Joystick] = None
        self.is_connected = False

        # Command rate limiting (Hz)
        self.wheel_update_rate = 60
        self.gimbal_update_rate = 60

        # Timing
        self.last_wheel_time = 0
        self.last_gimbal_time = 0
        self.last_button_time = 0
        self.button_debounce = 0.2  # 200ms button debounce

        # State tracking
        self.last_wheel_cmd = (0, 0)
        self.last_gimbal_cmd = (0, 0)
        self.last_cross_button = False
        self.last_circle_button = False
        self.last_square_button = False
        self.last_triangle_button = False

        # Moving average for smoother control
        self.wheel_buffer = deque(maxlen=2)

        # Event callbacks
        self.on_lights_toggle: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None

        # clock for frame timing 
        # TEMPORARILY COMMENTED OUT TO TRY DIFFERENT APPROACH
        # self.clock = pygame.time.Clock()

    def connect(self) -> bool:
        """Connect to DualSense Controller

        Returns:
            bool: True if connection success, False otherwise
        """
        if pygame.joystick.get_count() == 0:
            print("Error: No controller detected.")
            return False

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.is_connected = True

        print(f"Initialized: {self.joystick.get_name()} with ID {self.joystick.get_id()}, "
              f"axes: {self.joystick.get_numaxes()}, buttons: {self.joystick.get_numbuttons()}")

        # Initial vibration to confirm connection
        self._play_init_haptics()

        return True

    def disconnect(self):
        """Disconnect controller and stop all motors"""
        if self.is_connected and self.joystick:
            print("Disconnecting controller...")
            self.stop_all()
            self.joystick.stop_rumble()
            self.joystick.quit()
            self.is_connected = False

    def _play_init_haptics(self):
        """Play initial haptic feedback"""
        if not self.joystick:
            return
        try:
            self.joystick.rumble(0.3, 0.5, 50)
            time.sleep(0.15)
            self.joystick.stop_rumble()
            time.sleep(0.1)
            self.joystick.rumble(0.3, 0.5, 50)
            time.sleep(0.15)
            self.joystick.stop_rumble()
        except Exception as e:
            print(f"Error: Haptic feedback not available ({e})")

    def play_haptic(self, low_freq: float, high_freq: float, duration_ms: int):
        """Play haptic feedback

        Args:
            low_freq: Low frequency motor intensity (0.0 to 1.0)
            high_freq: High frequency motor intensity (0.0 to 1.0)
            duration_ms: Duration in milliseconds
        """
        if not self.joystick:
            return
        try:
            self.joystick.rumble(low_freq, high_freq, duration_ms)
        except Exception as e:
            print(f"Error: Haptic feedback not available ({e})")

    def apply_deadzone(self, value: float) -> float:
        """Apply deadzone with smooth transition

        Args:
            value: Joystick axis value (-1.0 to 1.0)
        Returns:
            float: Adjusted axis value with deadzone applied
        """
        if abs(value) < self.deadzone:
            return 0.0
        # Scale the output so deadzone doesnt create a jump
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.deadzone) / (1 - self.deadzone)

    def smooth_wheel_values(self, left: float, right: float) -> tuple:
        """
        Apply moving average for smoother wheel control
        Args:
            left: Left wheel speed
            right: Right wheel speed
        Returns:
            tuple: Smoothed (left, right) wheel speeds
        """
        self.wheel_buffer.append((left, right))
        if len(self.wheel_buffer) == 0:
            return 0, 0
        avg_left = sum(l for l, r in self.wheel_buffer) / len(self.wheel_buffer)
        avg_right = sum(r for l, r in self.wheel_buffer) / len(self.wheel_buffer)
        return round(avg_left, 3), round(avg_right, 3)

    def process_wheels(self, current_time: float):
        """Process wheel control with rate limiting

        Args:
            current_time: Current time in seconds
        """
        if current_time - self.last_wheel_time < 1.0 / self.wheel_update_rate:
            return

        # Read and process left stick
        left_x = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
        left_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_Y))

        # Calculate tank drive
        throttle = -left_y
        turn = -left_x

        left_speed = (throttle - turn) * self.max_wheel_speed
        right_speed = (throttle + turn) * self.max_wheel_speed

        # Clamp speeds
        left_speed = max(-self.max_wheel_speed, min(self.max_wheel_speed, left_speed))
        right_speed = max(-self.max_wheel_speed, min(self.max_wheel_speed, right_speed))

        # Apply smoothing
        left_speed, right_speed = self.smooth_wheel_values(left_speed, right_speed) # DISABLED FOR NOW

        # Only apply rate limiting to SENDING, not reading
        if current_time - self.last_wheel_time < 1.0 / self.wheel_update_rate:
            return


        # Send command if changed
        if (
            abs(left_speed - self.last_wheel_cmd[0]) > 0.005
            or abs(right_speed - self.last_wheel_cmd[1]) > 0.005
            or abs(left_speed) > 0.01
            or abs(right_speed) > 0.01
        ):
            self.base.base_speed_ctrl(left_speed, right_speed)
            self.last_wheel_cmd = (left_speed, right_speed)
        
        self.last_wheel_time = current_time

    def process_gimbal(self, current_time: float):
        """Process gimbal control with rate limiting
        Args:
            current_time: Current time in seconds
        """
        
        # Read and process right stick
        right_x = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_X))
        right_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_Y))

        # Convert to discrete commands
        x_input = 1 if right_x > 0.3 else (-1 if right_x < -0.3 else 0)
        y_input = -1 if right_y > 0.3 else (1 if right_y < -0.3 else 0)

        # Only apply rate limiting to SENDING, not reading
        if current_time - self.last_gimbal_time < 1.0 / self.gimbal_update_rate:
            return

        # Only send if changed
        if (x_input, y_input) != self.last_gimbal_cmd:
            if x_input == 0 and y_input == 0:
                self.base.send_command({"T": 135})
            else:
                self.base.gimbal_base_ctrl(x_input, y_input, self.gimbal_speed)
            self.last_gimbal_cmd = (x_input, y_input)
        
        self.last_gimbal_time = current_time

    def process_buttons(self, current_time: float) -> bool:
        """Process button inputs with debouncing
        Args:
            current_time: Current time in seconds
        Returns:
            bool: True to continue running, False to exit
        """
        # Square button for lights
        square_button = self.joystick.get_button(self.BUTTON_SQUARE)
        if square_button and not self.last_square_button:
            if current_time - self.last_button_time > self.button_debounce:
                self.base.base_lights_ctrl()
                if self.on_lights_toggle:
                    self.on_lights_toggle()
                self.last_button_time = current_time
        self.last_square_button = square_button

        # Triangle button for horn (haptic feedback)
        triangle_button = self.joystick.get_button(self.BUTTON_TRIANGLE)
        if triangle_button and not self.last_triangle_button:
            if current_time - self.last_button_time > self.button_debounce:
                self._play_init_haptics()
                audio_ctrl.play_audio_thread("sounds/double-car-horn.mp3")
                self.last_button_time = current_time
        self.last_triangle_button = triangle_button


        # Center gimbal on R1 press
        if self.joystick.get_button(self.BUTTON_R1):
            if current_time - self.last_button_time > self.button_debounce:
                self.center_gimbal()
                self.last_button_time = current_time
        # Options button to exit
        if self.joystick.get_button(self.BUTTON_OPTIONS):
            return False
        return True

    def update(self) -> bool:
        """Main update loop to be called regularly in main script

        Returns:
            bool: True to continue, False to exit
        """
        if not self.is_connected or not self.joystick:
            return False

        # Process events properly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        current_time = time.time()

        # Process controls with rate limiting
        self.process_wheels(current_time)
        self.process_gimbal(current_time)
        if not self.process_buttons(current_time):
            return False  # Exit signal

        # REMOVED. letting the main loop handle timing instead
        # self.clock.tick(60)  # 60 FPS main loop

        return True

    def stop_all(self):
        """Stop all motors and gimbal"""
        self.base.base_speed_ctrl(0, 0)
        self.base.send_command({"T": 135})  # Stop gimbal

    def center_gimbal(self):
        """Center gimbal position"""
        self.base.gimbal_ctrl(0, 0, 200, 100)


def test_controller():
    """Standalone test for DualSense controller"""
    print("DualSense Controller Test")
    print("-" * 40)

    import base_ctrl

    try:
        base = base_ctrl.BaseController("/dev/ttyTHS1", 115200)
        print("Base controller initialized")
    except Exception as e:
        print(f"Error: Failed to initialize base controller: {e}")
        return

    controller = DualSenseController(base)
    if not controller.connect():
        print("Error: Could not connect to controller.")
        return

    # Center gimbal
    print("Centering gimbal...")
    controller.center_gimbal()
    time.sleep(0.5)

    print("\n" + "=" * 40)
    print("Controller ready!")
    print("  Left stick: Wheels")
    print("  Right stick: Gimbal")
    print("  Cross (X): Toggle lights")
    print("  Options: Exit")
    print("=" * 40 + "\n")

    # main loop
    try:
        running = True
        while running:
            running = controller.update()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Stopping rover...")
        controller.stop_all()
        time.sleep(0.2)
        controller.disconnect()
        print("Test complete.")


if __name__ == "__main__":
    test_controller()
