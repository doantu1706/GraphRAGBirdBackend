import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

class WikipediaFetcher:
    def __init__(self):
        # Set User Agent để tuân thủ chính sách của Wikipedia
        # (Thư viện wikipedia python đôi khi bị chặn nếu không có header chuẩn, 
        # nhưng wrapper này thường tự xử lý, ta set lang mặc định là en trước)
        wikipedia.set_lang("en")

    def get_summary(self, bird_name: str, lang: str = 'vi') -> str:
        """
        Lấy tóm tắt về loài chim.
        Chiến lược:
        1. Thử tìm bằng ngôn ngữ yêu cầu (thường là 'vi').
        2. Nếu không thấy, thử tìm bằng tiếng Anh.
        3. Nếu gặp trang định hướng (Disambiguation), lấy kết quả đầu tiên.
        """
        
        # Thử với ngôn ngữ chính (Tiếng Việt)
        try:
            wikipedia.set_lang(lang)
            # search trước để lấy tên trang chính xác nhất
            search_results = wikipedia.search(bird_name)
            
            if not search_results:
                raise PageError(bird_name)
            
            # Lấy trang đầu tiên tìm được
            page = wikipedia.page(search_results[0])
            return page.summary[0:1000] # Lấy khoảng 1000 ký tự đầu
            
        except (PageError, DisambiguationError, Exception):
            # Fallback sang Tiếng Anh
            print(f"   [Wiki] '{bird_name}' not found in '{lang}'. Switching to English...")
            try:
                wikipedia.set_lang("en")
                search_results = wikipedia.search(bird_name)
                if search_results:
                    page = wikipedia.page(search_results[0])
                    return page.summary[0:1000]
            except Exception as e:
                print(f"   [Wiki Error] Could not fetch data: {e}")
        
        return None