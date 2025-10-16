import time
import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque

# import ollama

import audio_ctrl
import pygame
import base_ctrl

# from assistant_modules.control.dualsense_controller import DualSenseController
from assistant_modules.control.xbox360_controller import Xbox360Controller 

# def ask_llm(query):
#     response = ollama.chat(
#         model=LLM_MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Your name is Jetson, an autonomous rover that can interact with the world by accepting commands from the user. Only output pure raw text.",
#             },
#             {"role": "user", "content": query},
#         ],
#     )
#     return response["message"]["content"]

def main():
    print("Starting Main Program...")
    try:
        audio_ctrl.play_audio_thread("sounds/connected/ping.mp3")
    except Exception as e:
        print(f"Warning: Could not play startup sound: {e}")

    # Initialize base controller
    serial_port = "/dev/ttyTHS1"
    try:
        base = base_ctrl.BaseController(serial_port, 115200)
        print(f"Initialized: Base controller on {serial_port}")
    except Exception as e:
        print(
            f"Error initializing base controller: {e} \n Either {serial_port} is incorrect or the chassis is not connected."
        )
        return

    # Initialize pygame for joystick handling and audio
    # TODO: Implement a system to support multiple controller types
    pygame.init()
    controller = Xbox360Controller(
        base_controller=base,
        max_wheel_speed=0.2,  # Matches config.yaml max_speed
        gimbal_speed=50,  # Speed for gimbal UI control
        deadzone=0.15,  # Joystick deadzone
    )
    controller_active = controller.connect()

    # setup event callbacks maybe?
    def on_lights_toggle():
        print("Lights toggled " + ("ON" if base.base_light_status else "OFF"))
    controller.on_lights_toggle = on_lights_toggle # Assign the callback

    # center gimbal on start
    controller.center_gimbal()
    time.sleep(0.5)  # wait for gimbal to center

    # display controller info
    print("\n" + "=" * 40)
    print("Controller ready!")
    print("  Left stick: Wheels")
    print("  Right stick: Gimbal")
    print("  RB: Center Gimbal")
    print("  X: Toggle lights")
    print("  Start: Exit")
    print("=" * 40 + "\n")


    # ----- MAIN LOOP -----
    
    while True:
        try:
            if controller_active:
                controller_active = controller.update() # main loop status is basically controlled by the dualsense controller...
            else:
                print("Exiting: Controller disconnected...")
                return
            
            # insert ai shit here
            # itty bitty sleep to prevent cpu spinning?
            # time.sleep(0.001)  

        except KeyboardInterrupt:
            print("Exiting: User request...")
            if controller.is_connected:
                controller_active = False
                controller.disconnect()
            # exit main loop
            return
        except Exception as e:
            print(f"Exiting: Error in main loop: {e}")
            # Handle specific exceptions or errors
            if controller.is_connected:
                controller_active = False
                controller.disconnect()
            return

if __name__ == "__main__":
    # start_assistant()  # Commented out for controller testing
    main()  # Test controller gimbal support
