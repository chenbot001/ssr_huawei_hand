#!/usr/bin/env python3
import pyrealsense2 as rs
import time

def main():
    print("Testing RealSense T265 Connection...")
    
    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipeline = rs.pipeline()
    
    # Build config object and request pose data
    config = rs.config()
    config.enable_stream(rs.stream.pose)
    
    try:
        # Start streaming with requested config
        pipeline.start(config)
        print("Successfully connected to T265 and started pose stream.")
        print("Move the camera around to see the pose data change...")
        print("Press Ctrl+C to stop.\n")
        
        while True:
            # Wait for the next set of frames from the camera
            frames = pipeline.wait_for_frames()
            
            # Fetch pose frame
            pose_frame = frames.get_pose_frame()
            if pose_frame:
                pose_data = pose_frame.get_pose_data()
                
                # Get translation and rotation components
                t = pose_data.translation
                r = pose_data.rotation
                
                # Print output without scrolling the console too much
                print(f"\rPosition(x,y,z): [{t.x:6.3f}, {t.y:6.3f}, {t.z:6.3f}] | "
                      f"Rotation(x,y,z,w): [{r.x:6.3f}, {r.y:6.3f}, {r.z:6.3f}, {r.w:6.3f}]", end="")
                
            time.sleep(0.05)
            
    except Exception as e:
        print(f"\nFailed to connect or stream from T265: {e}")
        print("Please check if the T265 is plugged in correctly and no other programs are currently using it.")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        try:
            pipeline.stop()
            print("Pipeline stopped.")
        except:
            pass

if __name__ == "__main__":
    main()
