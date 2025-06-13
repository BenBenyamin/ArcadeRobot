## The code to run on the robot in order to estimate the latency of moving the arm. It outputs a csv file with the times to go from up->down and down->up.
## See video in the spring project update

import time
import csv
import statistics
import pygame
import stretch_body.robot

# === Arm positions in meters ===
DELTA = 0.015

# === Initialize Stretch robot node ===
robot = stretch_body.robot.Robot()
robot.startup()

# === Initialize pygame joystick ===
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(1)
joystick.init()

# === Latency storage ===
latency_log = []
csv_rows = [["command_time", "confirm_time", "target_pos", "latency_ms"]]

print("Instructions:")
print("1. Press ENTER to send the arm to a new position (UP/DOWN alternating).")
print("2. As soon as you visually SEE the arm finish moving, press Button 1 on the joystick to confirm.")
print("3. Type 'e' or 'E' and press ENTER at any prompt to exit.\n")

waiting_for_confirm = False
t_cmd = None
target_pos = None

robot.arm.move_by(-DELTA)
robot.push_command()

sign = 1.0
target_pos = {1.0: "UP", -1.0: "DOWN"}

running = True
while running:
    pygame.event.pump()

    # Step 1: Keyboard input to send motion command
    if not waiting_for_confirm:
        user_input = input("Press ENTER to send a motion command (or type 'e' to exit): ").strip().lower()
        if user_input == "e":
            print("Exit command received.")
            break

        t_cmd = time.time()
        robot.arm.move_by(2 * sign * DELTA)
        robot.push_command()
        print("Moving...")
        sign *= -1
        waiting_for_confirm = True

    stick_moved = abs(joystick.get_axis(4) - sign) <= 0.05

    # Step 2: Joystick movement to confirm motion completion
    if stick_moved and waiting_for_confirm:
        t_confirm = time.time()
        latency_ms = (t_confirm - t_cmd) * 1000
        print(f"[{t_confirm:.6f}] Motion visually confirmed. Latency: {latency_ms:.2f} ms")

        latency_log.append(latency_ms)
        csv_rows.append([f"{t_cmd:.6f}", f"{t_confirm:.6f}", target_pos[-sign], f"{latency_ms:.2f}"])
        waiting_for_confirm = False
        time.sleep(0.3)  # debounce

# === Final Reporting and Cleanup ===
if latency_log:
    mean_latency = statistics.mean(latency_log)
    std_latency = statistics.stdev(latency_log) if len(latency_log) > 1 else 0.0
    print(f"\nResults from {len(latency_log)} trials:")
    print(f"- Mean latency: {mean_latency:.2f} ms")
    print(f"- Std deviation: {std_latency:.2f} ms")
else:
    print("No latency data collected.")

# Save to CSV
filename = f"latency_log_{int(time.time())}.csv"
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)
    print(f"Latency log saved to {filename}")

pygame.quit()
