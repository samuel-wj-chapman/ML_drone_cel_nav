import os
from datetime import datetime

# Directory containing the images
image_directory = '../data/image_train_val'

# Define the cutoff date and time
cutoff_date = datetime.strptime('2024-11-11', '%Y-%m-%d').date()
cutoff_time = datetime.strptime('01:00:00', '%H:%M:%S').time()

# Iterate over all files in the directory
for filename in os.listdir(image_directory):
    # Extract the timestamp from the filename
    try:
        # Assuming the filename format is: latitude+longitude+timestamp.png
        parts = filename.split('+')
        timestamp_str = parts[2].split('.')[0]  # Extract the timestamp part
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')

        # Check if the date is November 11, 2024, and the time is later than 1 AM
        if timestamp.date() == cutoff_date and timestamp.time() > cutoff_time:
            # Construct the full file path
            file_path = os.path.join(image_directory, filename)
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {filename}")

    except (IndexError, ValueError) as e:
        print(f"Skipping file due to error: {filename} ({e})")