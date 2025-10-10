# vibe coded using claude sonnet 4.5...

import pygame
import base_ctrl
import time
from collections import deque

class DualSenseController:
    # DualSense axis mappings
    AXIS_LEFT_X = 0      # LS X
    AXIS_LEFT_Y = 1      # LS Y
    AXIS_LEFT_L2 = 2     # L2
    AXIS_RIGHT_X = 3     # RS X
    AXIS_RIGHT_Y = 4     # RS Y
    AXIS_RIGHT_L2 = 5    # R2
    AXIS_DPAD_X = 6      # DPAD X
    AXIS_DPAD_Y = 7      # DPAD Y

    # DualSense button mappings
    BUTTON_CROSS = 0      # X button
    BUTTON_CIRCLE = 1     # O button
    BUTTON_TRIANGLE = 2   # Triangle button
    BUTTON_SQUARE = 3     # Square button
    BUTTON_OPTIONS = 9    # Options button (START)
    
    def __init__(self, deadzone=0.05, gimbal_speed=50, max_wheel_speed=0.2):
        self.deadzone = deadzone
        self.gimbal_speed = gimbal_speed
        self.max_wheel_speed = max_wheel_speed
        
        # Command rate limiting (Hz)
        self.wheel_update_rate = 30  # 30Hz for wheels
        self.gimbal_update_rate = 20  # 20Hz for gimbal
        
        # Timing
        self.last_wheel_time = 0
        self.last_gimbal_time = 0
        self.last_button_time = 0
        self.button_debounce = 0.2  # 200ms button debounce
        
        # State tracking
        self.last_wheel_cmd = (0, 0)
        self.last_gimbal_cmd = (0, 0)
        self.last_cross_button = False
        

        
        # Moving average for smoother control
        self.wheel_buffer = deque(maxlen=3)
        
    def apply_deadzone(self, value):
        """Apply deadzone with smooth transition"""
        if abs(value) < self.deadzone:
            return 0.0
        # Scale the output so deadzone doesn't create a jump
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.deadzone) / (1 - self.deadzone)
    
    def smooth_wheel_values(self, left, right):
        """Apply moving average for smoother wheel control"""
        self.wheel_buffer.append((left, right))
        if len(self.wheel_buffer) == 0:
            return 0, 0
        avg_left = sum(l for l, r in self.wheel_buffer) / len(self.wheel_buffer)
        avg_right = sum(r for l, r in self.wheel_buffer) / len(self.wheel_buffer)
        return round(avg_left, 3), round(avg_right, 3)
    
    def process_wheels(self, joystick, base, current_time):
        """Process wheel control with rate limiting"""
        if current_time - self.last_wheel_time < 1.0 / self.wheel_update_rate:
            return
        
        # Read and process left stick
        left_x = self.apply_deadzone(joystick.get_axis(self.AXIS_LEFT_X))
        left_y = self.apply_deadzone(joystick.get_axis(self.AXIS_LEFT_Y))
        
        # Calculate tank drive
        throttle = -left_y
        turn = left_x
        
        left_speed = (throttle - turn) * self.max_wheel_speed
        right_speed = (throttle + turn) * self.max_wheel_speed
        
        # Clamp speeds
        left_speed = max(-self.max_wheel_speed, min(self.max_wheel_speed, left_speed))
        right_speed = max(-self.max_wheel_speed, min(self.max_wheel_speed, right_speed))
        
        # Apply smoothing
        left_speed, right_speed = self.smooth_wheel_values(left_speed, right_speed)
        
        # Only send if changed significantly (threshold of 0.01)
        if abs(left_speed - self.last_wheel_cmd[0]) > 0.01 or \
           abs(right_speed - self.last_wheel_cmd[1]) > 0.01:
            base.base_speed_ctrl(left_speed, right_speed)
            if left_speed != 0 or right_speed != 0:
                print(f"Wheels: L={left_speed:.2f}, R={right_speed:.2f}")
            self.last_wheel_cmd = (left_speed, right_speed)
            self.last_wheel_time = current_time
    
    def process_gimbal(self, joystick, base, current_time):
        """Process gimbal control with rate limiting"""
        if current_time - self.last_gimbal_time < 1.0 / self.gimbal_update_rate:
            return
        
        # Read and process right stick
        right_x = self.apply_deadzone(joystick.get_axis(self.AXIS_RIGHT_X))
        right_y = self.apply_deadzone(joystick.get_axis(self.AXIS_RIGHT_Y))
        
        # Convert to discrete commands
        x_input = 1 if right_x > 0.3 else (-1 if right_x < -0.3 else 0)
        y_input = -1 if right_y > 0.3 else (1 if right_y < -0.3 else 0)
        
        # Only send if changed
        if (x_input, y_input) != self.last_gimbal_cmd:
            if x_input == 0 and y_input == 0:
                base.send_command({"T": 135})
            else:
                print(f"Gimbal: X={x_input}, Y={y_input}")
                base.gimbal_base_ctrl(x_input, y_input, self.gimbal_speed)
            self.last_gimbal_cmd = (x_input, y_input)
            self.last_gimbal_time = current_time
    
    def process_buttons(self, joystick, base, current_time):
        """Process button inputs with debouncing"""
        # Cross button (X) for lights
        cross_button = joystick.get_button(self.BUTTON_CROSS)
        if cross_button and not self.last_cross_button:
            if current_time - self.last_button_time > self.button_debounce:
                base.base_lights_ctrl()
                print(f"Lights toggled " + ("ON" if base.base_light_status else "OFF"))
                self.last_button_time = current_time
        self.last_cross_button = cross_button
        
        # Options button to exit
        if joystick.get_button(self.BUTTON_OPTIONS):
            return False
        return True


def test_controller_gimbal():
    """Optimized DualSense controller test"""
    print("DualSense Controller Test Starting...")
    
    # Initialize pygame and controller
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("Error: No controller detected. Exiting.")
        return
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller '{joystick.get_name()}' initialized")
    print(f"  {joystick.get_numaxes()} axes, {joystick.get_numbuttons()} buttons")
    
    # Initial vibration to confirm connection
    joystick.rumble(0.3, 0.5, 50)
    time.sleep(0.15)
    joystick.stop_rumble()
    time.sleep(0.1)
    joystick.rumble(0.3, 0.5, 50)
    time.sleep(0.15)
    joystick.stop_rumble()

    
    # Initialize base controller
    try:
        base = base_ctrl.BaseController('/dev/ttyTHS1', 115200)
        print("Base controller initialized")
    except Exception as e:
        print(f"Error initializing base controller: {e}")
        return
    
    # Initialize controller handler
    controller = DualSenseController()
    
    # Center gimbal
    print("Centering gimbal...")
    base.gimbal_ctrl(0, 0, 0, 0)
    time.sleep(0.5)
    
    print("\nController ready!")
    print("Left stick: Control wheels")
    print("Right stick: Control gimbal")
    print("Cross (X): Toggle lights")
    print("Options: Exit\n")
    
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            # Process events properly
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            current_time = time.time()
            
            # Process controls with rate limiting
            controller.process_wheels(joystick, base, current_time)
            controller.process_gimbal(joystick, base, current_time)
            running = controller.process_buttons(joystick, base, current_time)
            
            # Maintain consistent frame rate
            clock.tick(60)  # 60 FPS main loop
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Stopping rover...")
        base.base_speed_ctrl(0, 0)
        base.send_command({"T": 135})
        time.sleep(0.2)
        print("Test complete.")


if __name__ == "__main__":
    test_controller_gimbal()