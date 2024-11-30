from importlib import reload
import cv2
import domino_detector  # Import the detection module
import traceback

def main():
    # Define the region to capture (adjust these values as needed)
    try:
        print("Press 'q' to quit.")
        
        region = {"top": 225, "left": 510, "width": 584, "height": 790}
        while True:
            try:
                # Capture a screenshot of the defined region
                frame = domino_detector.capture_screenshot(region)

                # Reload the domino_detector module to apply live changes
                reload(domino_detector)

                # Detect dominoes in the captured frame
                processed_frame = domino_detector.detect_dominoes(frame)
            except Exception as e:
                print(f"Domino detection error: {e}")
                processed_frame = frame

            # Display the processed frame
            cv2.imshow("Live Domino Detection", processed_frame)
            cv2.setWindowProperty(
                "Live Domino Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "Live Domino Detection", cv2.WINDOW_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Live Domino Detection", cv2.WND_PROP_TOPMOST, 1)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {traceback.format_exc()}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
