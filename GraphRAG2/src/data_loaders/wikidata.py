import sys
import unicodedata
import re
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataFetcher:
    def __init__(self):
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", "BirdGraphRAG/1.0 (contact@example.com)")

        # TỪ ĐIỂN CỨNG (Đã thêm nhiều biến thể)
        self.common_map = {
            # CHÀO MÀO
            "chim chào mào": "Pycnonotus jocosus",
            "chào mào": "Pycnonotus jocosus",
            "chào mào mũ": "Pycnonotus jocosus",
            "chào mào đít đỏ": "Pycnonotus jocosus",
            
            # BÓI CÁ
            "chim bói cá": "Alcedo atthis",
            "bói cá": "Alcedo atthis",
            
            # CÁC LOÀI KHÁC
            "chim sẻ": "Passer domesticus",
            "sẻ nhà": "Passer domesticus",
            "chích chòe": "Copsychus saularis",
            "chim sáo": "Acridotheres",
            "chim công": "Pavo cristatus",
            "đại bàng": "Aquila",
            "họa mi": "Garrulax canorus"
        }

    def _normalize_text(self, text):
        """Hàm chuẩn hóa chuỗi tiếng Việt siêu mạnh"""
        if not text: return ""
        # 1. Chuyển về chữ thường
        text = text.lower()
        # 2. Chuẩn hóa Unicode (NFC) -> Gom các dấu rời thành 1 ký tự
        text = unicodedata.normalize('NFC', text)
        # 3. Xóa khoảng trắng thừa (ví dụ "chim   chào  mào")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_bird_data(self, common_name: str):
        if not common_name: return None

        # --- BƯỚC 1: TRA TỪ ĐIỂN CỨNG ---
        normalized_name = self._normalize_text(common_name)
        
        # In ra log để xem nó đang so sánh cái gì
        print(f"      🔍 [Dict Check] Input: '{common_name}' -> Normalized: '{normalized_name}'")

        if normalized_name in self.common_map:
            search_term = self.common_map[normalized_name]
            print(f"      ✅ [Dict HIT] Found in dictionary: {search_term}")
        else:
            search_term = common_name
            print(f"      ⚠️ [Dict MISS] Searching Wikidata with raw name: {search_term}")

        # --- BƯỚC 2: GỌI WIKIDATA ---
        name_title = search_term.title()

        # Query vét cạn thông tin
        query = f"""
        SELECT ?scientificName ?image ?mass ?conservationLabel ?map ?wingspan ?lifespan ?foodLabel ?parentLabel WHERE {{
          {{ ?item rdfs:label "{search_term}"@vi. }}
          UNION {{ ?item rdfs:label "{search_term}"@en. }}
          UNION {{ ?item rdfs:label "{name_title}"@vi. }}
          UNION {{ ?item rdfs:label "{name_title}"@en. }}
          UNION {{ ?item wdt:P225 "{search_term}". }}
          
          ?item wdt:P225 ?scientificName.
          
          OPTIONAL {{ ?item wdt:P18 ?image. }}
          OPTIONAL {{ ?item wdt:P2067 ?mass. }}
          
          OPTIONAL {{ 
            ?item wdt:P141 ?statusItem. 
            ?statusItem rdfs:label ?conservationLabel.
            FILTER(LANG(?conservationLabel) = "vi") 
          }}
          
          OPTIONAL {{ ?item wdt:P181 ?map. }}
          OPTIONAL {{ ?item wdt:P2050 ?wingspan. }}
          OPTIONAL {{ ?item wdt:P2250 ?lifespan. }}
          
          OPTIONAL {{ 
            ?item wdt:P1034 ?food.
            ?food rdfs:label ?foodLabel.
            FILTER(LANG(?foodLabel) = "vi")
          }}
          
          OPTIONAL {{
            ?item wdt:P171 ?parent.
            ?parent rdfs:label ?parentLabel.
            FILTER(LANG(?parentLabel) = "vi")
          }}
        }}
        LIMIT 1
        """
        
        
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
            bindings = results["results"]["bindings"]
            
            if bindings:
                data = bindings[0]
                return {
                    "scientific_name": data["scientificName"]["value"],
                    "image_url": data.get("image", {}).get("value", None),
                    "mass": data.get("mass", {}).get("value", None),
                    "conservation": data.get("conservationLabel", {}).get("value", None),
                    "map_url": data.get("map", {}).get("value", None),
                    "wingspan": data.get("wingspan", {}).get("value", None),
                    "lifespan": data.get("lifespan", {}).get("value", None),
                    "food": data.get("foodLabel", {}).get("value", None),
                    "family": data.get("parentLabel", {}).get("value", None)
                }
            else:
                print(f"      ❌ [Wikidata] Found NOTHING for term: {search_term}")
                
        except Exception as e:
            print(f"      ❌ [Wikidata Error] {e}")
            
        return None
    
    def execute_generated_sparql(self, sparql_query: str):
        print(f"      🤖 [Wikidata] Executing AI-Generated Query...")
        
        self.sparql.setQuery(sparql_query)
        try:
            results = self.sparql.query().convert()
            bindings = results["results"]["bindings"]
            
            clean_results = []
            seen_names = set() # Tập hợp để kiểm tra trùng tên
            
            for item in bindings:
                # 1. Lấy tên (Xử lý lỗi thiếu key)
                name = "Unknown"
                if "birdLabel" in item: name = item["birdLabel"]["value"]
                elif "itemLabel" in item: name = item["itemLabel"]["value"]
                else:
                    for key in item.keys():
                        if "Label" in key: 
                            name = item[key]["value"]
                            break
                
                # 2. KIỂM TRA TRÙNG LẶP
                if name in seen_names:
                    continue # Bỏ qua nếu đã có con này rồi
                seen_names.add(name)

                # 3. Lấy ảnh
                image = item.get("image", {}).get("value", "")
                
                # 4. Lấy thông tin phụ
                extra_info = ""
                for key, val in item.items():
                    if key not in ["birdLabel", "itemLabel", "image", "bird", "item"] and "Label" not in key:
                        try:
                            # Làm tròn số
                            num = float(val['value'])
                            extra_info += f"{key.replace('mass', 'Nặng')}: {num:.2f} "
                        except:
                            extra_info += f"{val['value']} "
                
                clean_results.append({"name": name, "image": image, "info": extra_info.strip()})
                
            return clean_results
            
        except Exception as e:
            print(f"      ❌ [SPARQL Error] {e}")
            return []