import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import random
import piexif
import os
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# 🟢 قائمة بالأماكن العشوائية (إحداثيات جغرافية)
locations = [
    ("Toronto, Canada", 43.6532, -79.3832),
    ("Honolulu, Hawaii", 21.3069, -157.8583),
    ("Kathmandu, Nepal", 27.7172, 85.3240),
    ("Paris, France", 48.8566, 2.3522),
    ("Tokyo, Japan", 35.6895, 139.6917),
    ("Cairo, Egypt", 30.0444, 31.2357),
    ("Sydney, Australia", -33.8688, 151.2093),
]

# 🟢 تحميل الصورة وتحويلها إلى Tensor
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)

# 🔴 تطبيق هجوم عدائي
def adversarial_attack(image_tensor, epsilon=0.02):
    noise = torch.randn_like(image_tensor) * epsilon
    perturbed_image = image_tensor + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# 🔵 تحويل الصورة من Tensor إلى OpenCV
def tensor_to_cv2(tensor):
    image_np = tensor.squeeze().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# 🟣 تعديل بيانات EXIF وإضافة معلومات مضللة
def modify_exif(image_path):
    exif_dict = piexif.load(image_path) if piexif.load(image_path) else {"0th": {}, "Exif": {}, "GPS": {}}
    
    location_name, lat, lon = random.choice(locations)
    
    def convert_to_dms(value):
        d = int(value)
        m = int((value - d) * 60)
        s = int(((value - d) * 60 - m) * 60 * 100)
        return ((d, 1), (m, 1), (s, 100))
    
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = convert_to_dms(abs(lat))
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = convert_to_dms(abs(lon))
    
    exif_dict["0th"][piexif.ImageIFD.Make] = b"Anonymous Device"
    exif_dict["0th"][piexif.ImageIFD.Model] = b"Custom AI Processed"
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = b"2023:01:01 12:00:00"
    
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)
    
    return location_name

# 🔺 تنفيذ الهجوم العدائي وإضافة بيانات EXIF
def apply_attack_and_exif(image_path, epsilon=0.02):
    image_tensor = load_image(image_path)
    perturbed_image = adversarial_attack(image_tensor, epsilon)
    perturbed_cv2 = tensor_to_cv2(perturbed_image)
    
    processed_path = "modified_image.jpg"
    cv2.imwrite(processed_path, perturbed_cv2)
    
    location_name = modify_exif(processed_path)
    
    return processed_path, location_name

# 🟢 بوت تيليغرام
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # ضع التوكن هنا
bot = Bot(token=TOKEN)
app = Application.builder().token(TOKEN).build()

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("🔹 أهلاً بك! أرسل صورة وسأقوم بتعديلها وإخفاء بياناتها.")

async def handle_image(update: Update, context: CallbackContext) -> None:
    photo = update.message.photo[-1].get_file()
    photo_path = "received_image.jpg"
    await photo.download(photo_path)

    modified_path, location_name = apply_attack_and_exif(photo_path)
    await update.message.reply_text(f"✅ تم تعديل الصورة 📍 الموقع العشوائي: {location_name}")
    await update.message.reply_photo(photo=open(modified_path, "rb"))

    os.remove(photo_path)
    os.remove(modified_path)

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.PHOTO, handle_image))

print("🤖 البوت جاهز للعمل! أرسل صورة لبدء التعديل.")
app.run_polling()
