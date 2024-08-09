from pynput import mouse
import time
import json
import numpy as np

# Global data storage
baseline_data = []
current_session_data = []

def on_move(x, y):
    current_session_data.append({
        'type': 'move',
        'x': x,
        'y': y,
        'timestamp': time.time()
    })

def on_click(x, y, button, pressed):
    current_session_data.append({
        'type': 'click',
        'x': x,
        'y': y,
        'button': str(button),
        'pressed': pressed,
        'timestamp': time.time()
    })

def on_scroll(x, y, dx, dy):
    current_session_data.append({
        'type': 'scroll',
        'x': x,
        'y': y,
        'dx': dx,
        'dy': dy,
        'timestamp': time.time()
    })

def collect_data(duration=10):
    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        time.sleep(duration)
        listener.stop()

    print(f"Data collection complete. {len(current_session_data)} events captured.")

# Feature extraction functions

def calculate_speed(data):
    speeds = []
    for i in range(1, len(data)):
        if data[i]['type'] == 'move' and data[i-1]['type'] == 'move':
            dist = np.sqrt((data[i]['x'] - data[i-1]['x'])**2 + (data[i]['y'] - data[i-1]['y'])**2)
            time_diff = data[i]['timestamp'] - data[i-1]['timestamp']
            if time_diff > 0:
                speeds.append(dist / time_diff)
    return np.mean(speeds) if speeds else 0

def calculate_click_frequency(data):
    clicks = [event for event in data if event['type'] == 'click' and event['pressed']]
    if len(clicks) < 2:
        return 0
    intervals = [clicks[i]['timestamp'] - clicks[i-1]['timestamp'] for i in range(1, len(clicks))]
    return np.mean(intervals) if intervals else 0

def calculate_path_curvature(data):
    angles = []
    for i in range(2, len(data)):
        if data[i]['type'] == 'move' and data[i-1]['type'] == 'move' and data[i-2]['type'] == 'move':
            vector1 = (data[i-1]['x'] - data[i-2]['x'], data[i-1]['y'] - data[i-2]['y'])
            vector2 = (data[i]['x'] - data[i-1]['x'], data[i]['y'] - data[i-1]['y'])
            angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
            angles.append(abs(angle))
    return np.mean(angles) if angles else 0

def calculate_dwell_time(data):
    dwell_times = []
    for i in range(1, len(data)):
        if data[i]['type'] == 'move' and data[i-1]['type'] == 'move':
            if data[i]['x'] == data[i-1]['x'] and data[i]['y'] == data[i-1]['y']:
                dwell_times.append(data[i]['timestamp'] - data[i-1]['timestamp'])
    return np.mean(dwell_times) if dwell_times else 0

def calculate_idle_time(data):
    idle_times = []
    for i in range(1, len(data)):
        time_diff = data[i]['timestamp'] - data[i-1]['timestamp']
        if time_diff > 0.5:  # Considering more than 0.5s as idle time
            idle_times.append(time_diff)
    return np.mean(idle_times) if idle_times else 0

def calculate_dynamic_threshold(baseline_data, tolerance=1.5):
    metrics = ['speed', 'click_frequency', 'path_curvature', 'dwell_time', 'idle_time']
    thresholds = {}

    for metric in metrics:
        values = [globals()[f"calculate_{metric}"](session) for session in baseline_data]
        mean_value = np.mean(values)
        std_value = np.std(values)
        thresholds[metric] = (mean_value - tolerance * std_value, mean_value + tolerance * std_value)

    return thresholds

if __name__ == "__main__":
    # Step 1: Collect baseline data
    print("Starting baseline data collection...")
    collect_data(50)
    baseline_data.append(current_session_data.copy())
    current_session_data.clear()

     # Save the baseline data
    with open('baseline_mouse_data.json', 'w') as f:
        json.dump(baseline_data, f, indent=4)

    # Step 2: Calculate dynamic thresholds
    dynamic_thresholds = calculate_dynamic_threshold(baseline_data, tolerance=2.0)
    print(f"Dynamic Thresholds: {dynamic_thresholds}")

    # Step 3: Monitor real-time session
    print("Collecting real-time session data...")
    collect_data(10)

    # Feature calculation for the current session
    session_metrics = {
        'speed': calculate_speed(current_session_data),
        'click_frequency': calculate_click_frequency(current_session_data),
        'path_curvature': calculate_path_curvature(current_session_data),
        'dwell_time': calculate_dwell_time(current_session_data),
        'idle_time': calculate_idle_time(current_session_data),
    }

    print(f"Session Metrics: {session_metrics}")

    # Compare with dynamic thresholds
    matches = []
    for metric, value in session_metrics.items():
        lower, upper = dynamic_thresholds[metric]
        matches.append(lower - lower * 0.4 <= value <= upper + upper * 0.4)
        print(f"{metric}: {value:.2f} (Threshold: {(lower - lower * 0.4):.2f} - {(upper + upper * 0.4):.2f})")

    match_ratio = sum(matches) / len(matches)
    if match_ratio >= 0.6:  # Allow some tolerance (e.g., at least 60% of metrics must match)
        print("The user's behavior matches the expected pattern.")
    else:
        print("The user's behavior does not match the expected pattern.")
