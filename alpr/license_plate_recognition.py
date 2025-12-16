# FINAL WORKING VERSION - No OpenCV Display Issues
import cv2
import easyocr
import numpy as np
import os
import urllib.request
from datetime import datetime

print("=== FINAL ALPR System ===")
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

# 1. Create folder
if not os.path.exists('results'):
    os.makedirs('results')

# 2. Download detector
cascade_file = 'haarcascade_russian_plate_number.xml'
if not os.path.exists(cascade_file):
    print("Downloading plate detector...")
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml'
    urllib.request.urlretrieve(url, cascade_file)
    print("✅ Download completed")

# 3. Load image
img_path = 'car.jpg'
if not os.path.exists(img_path):
    print(f"❌ ERROR: Please add '{img_path}' to this folder")
    exit()

img = cv2.imread(img_path)
if img is None:
    print("❌ ERROR: Cannot read the image file")
    exit()

print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

# 4. Detect plate
plate_cascade = cv2.CascadeClassifier(cascade_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(60, 20))

print(f"\n🔍 Plate detection results:")
print(f"   Found {len(plates)} potential plate(s)")

# Filter small/false detections
valid_plates = []
for (x, y, w, h) in plates:
    area = w * h
    aspect_ratio = w / h if h > 0 else 0
    
    # License plates typically have aspect ratio > 2 (wide rectangle)
    if aspect_ratio > 2.0 and area > 2000:  # Minimum reasonable size
        valid_plates.append((x, y, w, h))
        print(f"   ✅ Valid plate: {w}x{h} (area: {area}, ratio: {aspect_ratio:.1f})")
    else:
        print(f"   ❌ False positive: {w}x{h} (area: {area}, ratio: {aspect_ratio:.1f})")

if len(valid_plates) == 0:
    print("\n⚠️ No valid plates found. Trying backup method...")
    height, width = img.shape[:2]
    # Try common plate position (bottom center)
    valid_plates = [[width//4, height-100, width//2, 80]]

# 5. Initialize OCR
print("\n📖 Initializing OCR (this may take a moment)...")
reader = easyocr.Reader(['en'])

# 6. Process each valid plate
all_detected_text = []
for i, (x, y, w, h) in enumerate(valid_plates):
    print(f"\n{'='*40}")
    print(f"🔧 Processing Plate {i+1}")
    print(f"   Position: ({x}, {y})")
    print(f"   Size: {w}x{h} pixels")
    print('='*40)
    
    # Crop plate region
    plate_roi = img[y:y+h, x:x+w]
    
    # Save original plate image
    plate_filename = f'results/plate_{i+1}_original.jpg'
    cv2.imwrite(plate_filename, plate_roi)
    print(f"   📸 Saved: {plate_filename}")
    
    # Preprocess for better OCR
    # Convert to grayscale
    if len(plate_roi.shape) == 3:
        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    else:
        plate_gray = plate_roi
    
    # Enhance contrast
    plate_gray = cv2.convertScaleAbs(plate_gray, alpha=1.5, beta=20)
    
    # Apply threshold
    _, plate_binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save processed plate
    processed_filename = f'results/plate_{i+1}_processed.jpg'
    cv2.imwrite(processed_filename, plate_binary)
    print(f"   🔧 Saved processed: {processed_filename}")
    
    # Perform OCR
    print("   📖 Reading text with OCR...")
    results = reader.readtext(plate_binary, paragraph=False, contrast_ths=0.1)
    
    if results:
        print("   ✅ Text detected:")
        for idx, (bbox, text, confidence) in enumerate(results):
            # Clean the text
            clean_text = ''.join(c.upper() for c in text if c.isalnum())
            
            if confidence > 0.1 and len(clean_text) >= 3:  # Lower threshold
                print(f"      Text {idx+1}: '{clean_text}'")
                print(f"      Confidence: {confidence:.2f}")
                
                # Add to results if it looks like a plate number
                if any(c.isdigit() for c in clean_text) and any(c.isalpha() for c in clean_text):
                    all_detected_text.append(clean_text)
                    
                    # Draw on image
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(img, clean_text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    else:
        print("   ❌ No text detected in this region")

# 7. Save final image
if all_detected_text:
    final_filename = 'results/final_result_with_plates.jpg'
    cv2.imwrite(final_filename, img)
    print(f"\n✅ Final image saved: {final_filename}")
    
    # Save text results
    text_filename = 'results/detected_plates.txt'
    with open(text_filename, 'w') as f:
        f.write("DETECTED LICENSE PLATES\n")
        f.write("=" * 30 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image: {img_path}\n")
        f.write("=" * 30 + "\n\n")
        for idx, plate in enumerate(all_detected_text, 1):
            f.write(f"{idx}. {plate}\n")
    
    print(f"📝 Text results saved: {text_filename}")
    
    print("\n🎯 FINAL DETECTED PLATE(S):")
    for plate in all_detected_text:
        print(f"   🚗 {plate}")
else:
    print("\n❌ No license plate text detected")

# 8. Summary
print(f"\n{'='*50}")
print("📊 SUMMARY")
print('='*50)
print(f"Total potential plates: {len(plates)}")
print(f"Valid plates: {len(valid_plates)}")
print(f"Text detections: {len(all_detected_text)}")
print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
print('='*50)

print("\n✅ Check the 'results' folder for all outputs!")
print("   - Original and processed plate images")
print("   - Final result image with bounding boxes")
print("   - Text file with detected plate numbers")