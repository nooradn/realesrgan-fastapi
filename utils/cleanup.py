"""
Cleanup functions for temporary files
"""
import modal
from core.config import app, web_image, temp_volume

# Auto cleanup function - runs every 30 minutes to delete expired files
@app.function(
    image=web_image,
    schedule=modal.Cron("*/30 * * * *"),  # Every 30 minutes
    timeout=300,  # 5 minute timeout to prevent hanging
    memory=512,  # Minimal memory for cleanup operations
    volumes={"/temp": temp_volume}
)
def cleanup_expired_files():
    """Delete files older than 1 hour (runs every 30 minutes)"""
    import os
    import time
    import glob
    
    try:
        print("🧹 Starting cleanup process...")
        
        # Reload volume to get latest state
        temp_volume.reload()
        
        current_time = int(time.time())
        deleted_count = 0
        processed_count = 0
        
        # Find all image files in temp directory (PNG and JPG)
        print("🔍 Scanning for image files...")
        temp_files = []
        
        # Use more specific patterns to avoid issues
        for pattern in ["/temp/*.png", "/temp/*.jpg"]:
            files = glob.glob(pattern)
            temp_files.extend(files)
            print(f"📁 Pattern {pattern}: found {len(files)} files")
        
        print(f"📊 Total files to process: {len(temp_files)}")
        
        if not temp_files:
            print("✅ No files found in temp directory")
            return "No files to cleanup"
        
        for file_path in temp_files:
            try:
                processed_count += 1
                filename = os.path.basename(file_path)
                
                # Progress indicator for long operations
                if processed_count % 10 == 0:
                    print(f"📈 Processed {processed_count}/{len(temp_files)} files...")
                
                # Extract timestamp from filename: {uuid}_{timestamp}.ext
                if "_" not in filename:
                    print(f"⚠️ Skipping file with invalid format: {filename}")
                    continue
                
                parts = filename.split("_")
                if len(parts) < 2:
                    print(f"⚠️ Skipping file with invalid format: {filename}")
                    continue
                
                timestamp_part = parts[1].split(".")[0]
                
                try:
                    file_timestamp = int(timestamp_part)
                except ValueError:
                    print(f"⚠️ Invalid timestamp in filename: {filename}")
                    continue
                
                # Check if file is older than 1 hour (3600 seconds) - files expire after 1 hour
                age_seconds = current_time - file_timestamp
                
                if age_seconds > 3600:
                    # Verify file exists before deletion
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"🗑️ Deleted expired file: {filename} (age: {age_seconds//60} minutes)")
                    else:
                        print(f"⚠️ File already gone: {filename}")
                else:
                    print(f"⏰ Keeping file: {filename} (age: {age_seconds//60} minutes)")
                    
            except Exception as e:
                print(f"❌ Error processing file {file_path}: {str(e)}")
                continue
        
        print(f"📊 Cleanup summary: processed {processed_count}, deleted {deleted_count}")
        
        # Only commit if we actually deleted files
        if deleted_count > 0:
            print("💾 Committing volume changes...")
            temp_volume.commit()
            print(f"✅ Cleanup complete: {deleted_count} expired files deleted")
        else:
            print("✅ Cleanup complete: No expired files found")
        
        return f"Processed {processed_count} files, deleted {deleted_count} expired files"
        
    except Exception as e:
        error_msg = f"❌ Cleanup failed: {str(e)}"
        print(error_msg)
        return error_msg