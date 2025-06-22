from PIL import Image
import google.generativeai as genai
import os
from tkinter import Tk, filedialog

# ========== Step 1: File Picker ==========
def choose_image_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image Files", "*.bmp *.jpg *.jpeg *.png")]
    )
    return file_path

# ========== Step 2: Convert BMP to JPG ==========
def convert_bmp_to_jpg(bmp_path):
    jpg_path = os.path.splitext(bmp_path)[0] + ".jpg"
    try:
        with Image.open(bmp_path) as bmp_image:
            rgb_image = bmp_image.convert('RGB')
            rgb_image.save(jpg_path, 'JPEG')
            print(f"[✓] Converted: {bmp_path} --> {jpg_path}")
        return jpg_path
    except Exception as e:
        print(f"[X] Error during conversion: {e}")
        return None

# ========== Step 3: Analyze Image Using Gemini API ==========
def analyze_image_with_gemini(api_key, image_path):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        with Image.open(image_path) as img:
            response = model.generate_content(
                [
                    "Determine if this is a fingerprint image. Answer strictly 'Yes' or 'No' and provide a brief reason.",
                    img
                ],
                stream=False
            )
        print("\n[✓] Gemini Analysis Result:")
        print(response.text.strip())

    except Exception as e:
        print(f"[X] Gemini API Error: {e}")

# ========== Main ==========
if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyADiuoPIC_LhcTANQavgWdJ5aNsK-C86BE")
    
    image_path = choose_image_file()
    
    if not image_path:
        print("[X] No file selected.")
    elif not GEMINI_API_KEY:
        print("[X] Gemini API key is missing. Set it using environment variable 'GEMINI_API_KEY'.")
    else:
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".bmp":
            image_path = convert_bmp_to_jpg(image_path)
        if image_path:  # Only proceed if conversion succeeded or not needed
            analyze_image_with_gemini(GEMINI_API_KEY, image_path)
