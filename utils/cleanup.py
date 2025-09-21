"""
Cleanup functions for temporary files
"""
import modal
from core.config import app, web_image, temp_volume

# Auto cleanup function - runs every hour to delete expired files
@app.function(
    image=web_image,
    schedule=modal.Cron("0 * * * *"),  # Every hour at minute 0
    volumes={"/temp": temp_volume}
)
def cleanup_expired_files():
    """Delete files older than 1 hour"""
    import os
    import time
    import glob
    
    current_time = int(time.time())
    deleted_count = 0
    
    # Find all PNG files in temp directory
    temp_files = glob.glob("/temp/*.png")
    
    for file_path in temp_files:
        try:
            # Extract timestamp from filename: {uuid}_{timestamp}.png
            filename = os.path.basename(file_path)
            if "_" in filename:
                timestamp_str = filename.split("_")[1].split(".")[0]
                file_timestamp = int(timestamp_str)
                
                # Check if file is older than 1 hour (3600 seconds)
                if current_time - file_timestamp > 3600:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"🗑️ Deleted expired file: {filename}")
                    
        except (ValueError, IndexError, OSError) as e:
            print(f"⚠️ Error processing file {file_path}: {e}")
            continue
    
    # Commit changes to volume
    if deleted_count > 0:
        temp_volume.commit()
        print(f"✅ Cleanup complete: {deleted_count} expired files deleted")
    else:
        print("✅ Cleanup complete: No expired files found")
    
    return f"Deleted {deleted_count} expired files"