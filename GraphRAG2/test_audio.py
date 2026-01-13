# File: test_audio.py
from src.data_loaders.xenocanto import XenoCantoFetcher

print("--- BẮT ĐẦU TEST (SCRAPING MODE) ---")

sci_name = "Alcedo atthis" 
fetcher = XenoCantoFetcher()

print(f"📡 Đang cào dữ liệu cho: {sci_name}")

try:
    result = fetcher.get_audio(sci_name)
    
    if result:
        print(f"✅ THÀNH CÔNG!")
        print(f"🎵 URL: {result['url']}")
        print(f"📍 Location: {result['loc']}")
    else:
        print("❌ Không tìm thấy dữ liệu.")

except Exception as e:
    print(f"❌ Lỗi: {e}")