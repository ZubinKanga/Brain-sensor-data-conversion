from pylsl import StreamInfo, StreamOutlet
from pynput import keyboard

# Create LSL info stream for char markers
info = StreamInfo('SSVEPKeyMarkers', 'Markers', 1, 0, 'string', 'SSVEPKeyMarkers')

# Make an outlet
outlet = StreamOutlet(info)

# Function to capture key release and send LSL marker
def on_release(key):
    # Convert to string
    strkey = "{0}".format(key)
    print('Key released: ', strkey)

    # Send as LSL marker
    outlet.push_sample([strkey])

    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_release=on_release) as listener:
    listener.join()