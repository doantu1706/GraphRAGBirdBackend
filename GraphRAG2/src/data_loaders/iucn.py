import requests
import os

class IUCNFetcher:
    def __init__(self):
        # Lấy token từ biến môi trường (cấu hình trong .env)
        self.token = os.getenv("IUCN_API_TOKEN", "YOUR_TOKEN_HERE")
        self.base_url = "https://apiv3.iucnredlist.org/api/v3"
        
        # Bảng map mã code sang tên đầy đủ cho dễ đọc
        self.category_map = {
            "EX": "Extinct (Tuyệt chủng)",
            "EW": "Extinct in the Wild (Tuyệt chủng trong tự nhiên)",
            "CR": "Critically Endangered (Cực kỳ nguy cấp)",
            "EN": "Endangered (Nguy cấp)",
            "VU": "Vulnerable (Sắp nguy cấp)",
            "NT": "Near Threatened (Sắp bị đe dọa)",
            "LC": "Least Concern (Ít quan tâm)",
            "DD": "Data Deficient (Thiếu dữ liệu)",
            "NE": "Not Evaluated (Chưa đánh giá)"
        }

    def get_conservation_status(self, scientific_name: str) -> str:
        """
        Lấy trạng thái bảo tồn dựa trên Tên Khoa Học.
        """
        if not scientific_name:
            return None

        # Kiểm tra nếu chưa có Token thật
        if self.token == "YOUR_TOKEN_HERE" or not self.token:
            print("   [IUCN Warning] Missing API Token. Using Mock/Wikidata is recommended instead.")
            return "Unknown (Missing API Token)"

        # Gọi API IUCN
        try:
            # Endpoint: /species/{name}?token={token}
            url = f"{self.base_url}/species/{scientific_name}?token={self.token}"
            response = requests.get(url, timeout=10)
            data = response.json()

            # Kiểm tra kết quả
            if 'result' in data and len(data['result']) > 0:
                # Lấy category code (ví dụ: 'LC', 'EN')
                code = data['result'][0].get('category', 'NE')
                # Map sang tên đầy đủ
                return self.category_map.get(code, code)
            else:
                return "Not Found in IUCN Database"

        except Exception as e:
            print(f"   [IUCN Error] {e}")
            return None

# --- Mẹo nhỏ (Pro Tip) ---
# Nếu bạn không muốn xin API Key của IUCN (vì duyệt khá lâu),
# bạn có thể dùng WikidataFetcher để lấy trạng thái bảo tồn.
# Trong Wikidata, Property P141 chính là "IUCN conservation status".