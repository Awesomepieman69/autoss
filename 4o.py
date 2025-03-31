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
import re

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

# --- REMOVED parse_questions_from_text function ---
# (The previous function definition is deleted)

# Query OpenAI for answers from the full text block
def get_openai_answer(full_text_block):
    """Sends the full text block from ROI to OpenAI, asking for all answers."""
    
    print(f"Sending full text block (length: {len(full_text_block)}) to OpenAI...")

    messages = [
        {"role": "system", "content": "You are an expert test-taking assistant. Analyze the following text block which contains one or more questions, potentially with options. For EACH question you identify (e.g., Question 1, Question 2, etc.), determine the single best answer or option based ONLY on the text provided. Format your entire response as a list, with each line containing the question identifier (like \"Question 1\" or \"1.\") followed by a colon and the single correct answer identifier or text (e.g., A, B, 1, 2, True, False, or the answer text itself). Do NOT include explanations or any other text. Example response format:\nQuestion 1: A\nQuestion 2: True\n3.: Option Text\nQuestion 4: B"},
        {"role": "user", "content": full_text_block}
    ]
    
    try:
        # Ensure the openai client library is compatible with this create call if using v1.x+
        # Might need: response = openai.chat.completions.create(...) for newer library versions
        response = openai.ChatCompletion.create(
            model="gpt-4o", 
            messages=messages,
            temperature=0 
        )
        answer = response["choices"][0]["message"]["content"].strip()
        print(f"OpenAI response for full block:\n{answer}")
        return answer
    except Exception as e:
        # Add more specific error checking if needed (e.g., rate limits, auth errors)
        print(f"Error calling OpenAI for full block: {e}")
        return "OpenAI Error"

# GUI Application
class ScreenCaptureApp:
    # ... (keep __init__, create_widgets, select_roi, display_image, toggle_capture)
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Screen Capture")
        self.root.geometry("800x600")
        self.monitor_index = 1
        self.roi = None
        self.capturing = False
        self.capture_thread = None
        self.create_widgets()
        
    def create_widgets(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(control_frame, text="Monitor:").pack(side=tk.LEFT, padx=5)
        self.monitor_var = tk.StringVar(value="1")
        monitor_entry = ttk.Entry(control_frame, textvariable=self.monitor_var, width=5)
        monitor_entry.pack(side=tk.LEFT, padx=5)
        self.roi_btn = ttk.Button(control_frame, text="Select ROI", command=self.select_roi)
        self.roi_btn.pack(side=tk.LEFT, padx=5)
        self.start_stop_btn = ttk.Button(control_frame, text="Start Capture", command=self.toggle_capture)
        self.start_stop_btn.pack(side=tk.LEFT, padx=5)
        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text_output = scrolledtext.ScrolledText(text_frame, height=10)
        self.text_output.pack(fill=tk.BOTH, expand=True)
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
        monitor_image = capture_monitor_macos(monitor_index=self.monitor_index)
        selected_roi = select_roi_opencv(monitor_image.copy())
        if selected_roi:
            self.roi = selected_roi
            x, y, w, h = self.roi
            self.status_var.set(f"ROI selected: x={x}, y={y}, w={w}, h={h}. Press Start Capture.")
            highlight_image = monitor_image.copy()
            cv2.rectangle(highlight_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.display_image(highlight_image)
        else:
            self.roi = None
            self.status_var.set("ROI selection cancelled or invalid.")
            
    def display_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        if label_width > 0 and label_height > 0:
            img_width, img_height = pil_image.size
            scale = min(label_width/img_width, label_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.tk_image)
        
    def toggle_capture(self):
        if self.capturing:
            self.capturing = False
            self.start_stop_btn.config(text="Start Capture")
            self.status_var.set("Capture stopped")
            self.roi_btn.config(state=tk.NORMAL)
        else:
            if not self.roi:
                self.status_var.set("Please select ROI first")
                return
            self.capturing = True
            self.start_stop_btn.config(text="Stop Capture")
            self.status_var.set("Capturing...")
            self.roi_btn.config(state=tk.DISABLED)
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
    # Updated capture_loop
    def capture_loop(self):
        while self.capturing:
            try:
                # Capture screen, Extract ROI, Display Image
                monitor_image = capture_monitor_macos(monitor_index=self.monitor_index)
                x, y, w, h = self.roi
                selected_image = monitor_image[y:y+h, x:x+w]
                self.root.after(0, lambda img=selected_image.copy(): self.display_image(img))

                # 1. Extract raw text from the ROI
                print("\n--- Extracting text from ROI ---")
                raw_extracted_text = extract_text(selected_image) 
                print(f"Raw OCR Text Length: {len(raw_extracted_text)}")
                if not raw_extracted_text:
                     print("OCR returned no text.")
                     self.root.after(0, lambda: self.status_var.set("No text detected in ROI"))
                     time.sleep(5)
                     continue # Skip to next iteration if no text
                print("--------------------------------")

                # 2. Send the *entire* raw text block to OpenAI
                print(f"Sending full text block (length {len(raw_extracted_text)}) to OpenAI...")
                answer_block = get_openai_answer(raw_extracted_text) # Pass the full text
                
                # 3. Update GUI with the multi-answer response block
                timestamp = time.strftime("%H:%M:%S")
                self.root.after(0, lambda ts=timestamp, ans=answer_block: self.update_text_output(ts, ans))
                
                # Wait for 5 seconds
                time.sleep(5)
                
            except Exception as e:
                error_msg = f"Error in capture loop: {type(e).__name__} - {e}"
                print(error_msg) 
                import traceback
                traceback.print_exc() 
                self.root.after(0, lambda err=error_msg: self.status_var.set(err))
                time.sleep(5)
    
    # Updated update_text_output
    def update_text_output(self, timestamp, answer_block):
        # answer_block is now the potentially multi-line string response from OpenAI
        self.text_output.insert(tk.END, f"--- Processed at {timestamp} ---\n")
        self.text_output.insert(tk.END, f"{answer_block}\n") # Insert the whole block
        self.text_output.insert(tk.END, "\n")
        self.text_output.see(tk.END)
        self.status_var.set(f"Processed at {timestamp}")

# Main function
# ... (keep main)
def main():
    screens = get_monitors()
    if screens is not None and len(screens) == 0:
        return
    root = tk.Tk()
    app = ScreenCaptureApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
