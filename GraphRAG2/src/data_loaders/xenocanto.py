from urllib.parse import quote

class XenoCantoFetcher:
    def get_audio(self, scientific_name: str):
        if not scientific_name: return None
        
        # Tạo đường link tìm kiếm trực tiếp trên trang chủ Xeno-canto
        # Ví dụ: https://xeno-canto.org/explore?query=Alcedo%20atthis
        encoded_name = quote(scientific_name)
        search_url = f"https://xeno-canto.org/explore?query={encoded_name}"
        
        print(f"      🔗 Generated Safe Link: {search_url}")
        
        # Trả về link trang web thay vì link file .mp3
        return {
            "url": search_url,
            "loc": "Xeno-canto Database", # Địa điểm: Kho dữ liệu
            "rec": "Search Link"          # Người thu: Link tìm kiếm
        }