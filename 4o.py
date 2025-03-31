import os
import cv2
import numpy as np
import pytesseract
import openai
import subprocess
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY not set. Please set it in a .env file.")

# Get list of monitors using AppKit on macOS
def get_monitors():
    try:
        from AppKit import NSScreen
        screens = NSScreen.screens()
        if len(screens) == 0:
            print("No monitor detected. Exiting.")
            return []
        else:
            print(f"Detected {len(screens)} monitor(s).")
            return screens
    except ImportError:
        print("AppKit module not available. Cannot verify monitor connection. Proceeding with default monitor.")
        return None

# Screen capture using macOS screencapture with monitor selection (-D flag)
def capture_monitor_macos(monitor_index=1, output_path="/tmp/screenshot.png"):
    # Use the -D flag to specify which monitor to capture from.
    # The monitor_index here is expected to be 1-indexed.
    subprocess.run(["screencapture", "-x", "-D", str(monitor_index), output_path], check=True)
    image = cv2.imread(output_path)
    return image

# Custom ROI selection with green rectangle and immediate window closure
def select_roi_opencv(image):
    # Resize image to 50% for smoother selection (adjust scale_factor if needed)
    scale_factor = 0.5
    small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # Variables for ROI selection
    selecting = False
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    roi = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_x, start_y, end_x, end_y, roi, small_image
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_x, start_y = x, y
            end_x, end_y = x, y
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if selecting:
                end_x, end_y = x, y
        
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            end_x, end_y = x, y
            # Scale coordinates back to original image size
            x_original = int(min(start_x, end_x) / scale_factor)
            y_original = int(min(start_y, end_y) / scale_factor)
            w_original = int(abs(end_x - start_x) / scale_factor)
            h_original = int(abs(end_y - start_y) / scale_factor)
            roi = (x_original, y_original, w_original, h_original)

    # Create window and set mouse callback
    window_name = "Select ROI (Press Enter to confirm, Esc to cancel)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # Create a copy of the small image to draw on
        display_image = small_image.copy()
        if selecting or (start_x != -1 and end_x != -1):
            # Draw green rectangle
            cv2.rectangle(display_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            if roi is not None:
                break
        elif key == 27:  # Esc key
            roi = None
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)  # Ensure immediate closure
    return roi

# Image preprocessing for OCR
def preprocess_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Extract text with Tesseract
def extract_text(image):
    height, width = image.shape[:2]
    if max(height, width) > 1000:
        scale = 1000 / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    preprocessed = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(preprocessed, lang='eng', config='--psm 6')
    return " ".join(text.split()).strip()

# Query OpenAI for an answer based on the provided text
def get_openai_answer(text):
    messages = [
        {"role": "system", "content": "You are an expert tutor that provides accurate, clear, and concise answers to questions. Your answers should be straightforward and focused on providing the correct answer with a brief explanation of why it is correct if you see options just tell me correct one same goes for blank MCQs"},
        {"role": "user", "content": text}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )
    return response["choices"][0]["message"]["content"].strip()

# GUI Application
class ScreenCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Screen Capture")
        self.root.geometry("800x600")
        
        # Variables
        self.monitor_index = 1
        self.roi = None
        self.capturing = False
        self.capture_thread = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Monitor selection
        ttk.Label(control_frame, text="Monitor:").pack(side=tk.LEFT, padx=5)
        self.monitor_var = tk.StringVar(value="1")
        monitor_entry = ttk.Entry(control_frame, textvariable=self.monitor_var, width=5)
        monitor_entry.pack(side=tk.LEFT, padx=5)
        
        # ROI selection button
        self.roi_btn = ttk.Button(control_frame, text="Select ROI", command=self.select_roi)
        self.roi_btn.pack(side=tk.LEFT, padx=5)
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(control_frame, text="Start Capture", command=self.toggle_capture)
        self.start_stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Middle frame for image display
        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Image display
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame for text output
        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text output
        self.text_output = scrolledtext.ScrolledText(text_frame, height=10)
        self.text_output.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
    
    def select_roi(self):
        try:
            monitor_index = int(self.monitor_var.get())
            if monitor_index < 1:
                monitor_index = 1
        except ValueError:
            monitor_index = 1

        self.monitor_index = monitor_index
        self.status_var.set(f"Capturing monitor {monitor_index} for ROI selection...")
        self.root.update()

        # Capture screen
        monitor_image = capture_monitor_macos(monitor_index=self.monitor_index)

        # Select ROI using the separate OpenCV window
        selected_roi = select_roi_opencv(monitor_image.copy()) # Use a copy for selection

        if selected_roi:
            self.roi = selected_roi # Store the selected ROI
            x, y, w, h = self.roi
            self.status_var.set(f"ROI selected: x={x}, y={y}, w={w}, h={h}. Press Start Capture.")

            # Draw the selected ROI rectangle on the original monitor image
            highlight_image = monitor_image.copy()
            cv2.rectangle(highlight_image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle

            # Display the full monitor image with the ROI highlighted in the GUI
            self.display_image(highlight_image)
        else:
            self.roi = None # Clear ROI if selection was cancelled
            self.status_var.set("ROI selection cancelled or invalid.")
            # Optionally clear the image display or show the original monitor image without ROI
            # self.display_image(monitor_image) # Or clear it
            # For now, we'll leave the last displayed image
    
    def display_image(self, cv_image):
        # Convert OpenCV image to PIL format
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Resize to fit the label while maintaining aspect ratio
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        if label_width > 0 and label_height > 0:
            # Calculate scaling factor to fit the label
            img_width, img_height = pil_image.size
            scale = min(label_width/img_width, label_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize the image
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage and display
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.tk_image)
    
    def toggle_capture(self):
        if self.capturing:
            # Stop capturing
            self.capturing = False
            self.start_stop_btn.config(text="Start Capture")
            self.status_var.set("Capture stopped")
            self.roi_btn.config(state=tk.NORMAL)
        else:
            # Start capturing
            if not self.roi:
                self.status_var.set("Please select ROI first")
                return
            
            self.capturing = True
            self.start_stop_btn.config(text="Stop Capture")
            self.status_var.set("Capturing...")
            self.roi_btn.config(state=tk.DISABLED)
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
    
    def capture_loop(self):
        while self.capturing:
            try:
                # Capture screen
                monitor_image = capture_monitor_macos(monitor_index=self.monitor_index)
                
                # Extract ROI
                x, y, w, h = self.roi
                selected_image = monitor_image[y:y+h, x:x+w]
                
                # Process image
                extracted_text = extract_text(selected_image)
                
                # Update UI
                self.root.after(0, lambda img=selected_image: self.display_image(img))
                
                if extracted_text:
                    # Get answer from OpenAI
                    answer = get_openai_answer(extracted_text)
                    timestamp = time.strftime("%H:%M:%S")
                    
                    # Update text output
                    self.root.after(0, lambda t=extracted_text, a=answer, ts=timestamp: self.update_text_output(ts, t, a))
                else:
                    self.root.after(0, lambda: self.status_var.set("No text detected"))
                
                # Wait for 5 seconds
                time.sleep(5)
                
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.status_var.set(f"Error: {err}"))
                time.sleep(5)
    
    def update_text_output(self, timestamp, text, answer):
        self.text_output.insert(tk.END, f"[{timestamp}] Detected: {text}\n")
        self.text_output.insert(tk.END, f"Answer: {answer}\n\n")
        self.text_output.see(tk.END)
        self.status_var.set(f"Processed at {timestamp}")

# Main function
def main():
    screens = get_monitors()
    if screens is not None and len(screens) == 0:
        return
    
    # Create Tkinter application
    root = tk.Tk()
    app = ScreenCaptureApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
