from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config

class Neo4jHandler:
    def __init__(self):
        # Kết nối Neo4j
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        
        print("   ⏳ Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self._init_indices()

    def _init_indices(self):
        # Tạo Vector Index với 384 dimensions
        query = """
        CREATE VECTOR INDEX bird_desc_index IF NOT EXISTS
        FOR (w:WikiInfo) ON (w.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 384, 
            `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)

    def close(self):
        self.driver.close()

    def check_data_status(self, scientific_name):
        """Kiểm tra xem dữ liệu đã có những gì"""
        query = """
        MATCH (b:Bird {scientific_name: $sci})
        RETURN b,
               EXISTS((b)-[:HAS_INFO]->(:WikiInfo)) as has_wiki,
               EXISTS((b)-[:HAS_SOUND]->(:Audio)) as has_audio,
               EXISTS((b)-[:HAS_ECOLOGY]->(:Ecology)) as has_ecology,
               EXISTS((b)-[:HAS_STATUS]->(:IUCN)) as has_status,
               b.image_url IS NOT NULL as has_image, 
               b.mass IS NOT NULL as has_mass
        """
        with self.driver.session() as session:
            res = session.run(query, sci=scientific_name).single()
            
            if not res:
                return {"exists": False}
            
            return {
                "exists": True,
                "common_name": res['b'].get('common_name'),
                "has_wiki": res['has_wiki'],
                "has_audio": res['has_audio'],
                "has_status": res['has_status'],
                "has_ecology": res['has_ecology'],
                "has_image": res['has_image'],
                "has_mass": res['has_mass']
            }

    # --- HÀM ĐÃ ĐƯỢC NÂNG CẤP ĐỂ NHẬN NHIỀU THAM SỐ ---
    def update_details(self, scientific_name, image_url=None, mass=None, map_url=None, wingspan=None, lifespan=None, food=None, family=None, conservation=None):
        """
        Lưu tất cả thông tin chi tiết từ Wikidata vào node Bird.
        Các tham số mặc định là None để tránh lỗi nếu thiếu dữ liệu.
        """
        # Nếu không có thông tin gì mới thì thoát luôn
        if not any([image_url, mass, map_url, wingspan, lifespan, food, family, conservation]):
            return

        query = """
        MERGE (b:Bird {scientific_name: $sci})
        SET b.image_url = COALESCE($image, b.image_url),
            b.mass = COALESCE($mass, b.mass),
            b.map_url = COALESCE($map, b.map_url),         // Thêm bản đồ
            b.wingspan = COALESCE($wingspan, b.wingspan),   // Thêm sải cánh
            b.lifespan = COALESCE($lifespan, b.lifespan),   // Thêm tuổi thọ
            b.food = COALESCE($food, b.food),               // Thêm thức ăn
            b.family = COALESCE($family, b.family),         // Thêm họ
            b.conservation = COALESCE($conservation, b.conservation) // Thêm bảo tồn
        """
        with self.driver.session() as session:
            session.run(query, sci=scientific_name, 
                        image=image_url, mass=mass, map=map_url, 
                        wingspan=wingspan, lifespan=lifespan, 
                        food=food, family=family, conservation=conservation)

    def update_wiki(self, scientific_name, common_name, summary):
        if not summary: return
        vector = self.embeddings.embed_query(summary)
        query = """
        MERGE (b:Bird {scientific_name: $sci})
        SET b.common_name = COALESCE(b.common_name, $common)
        MERGE (w:WikiInfo {bird_id: $sci})
        SET w.summary = $summary, w.embedding = $vector
        MERGE (b)-[:HAS_INFO]->(w)
        """
        with self.driver.session() as session:
            session.run(query, sci=scientific_name, common=common_name, summary=summary, vector=vector)

    def update_audio(self, scientific_name, audio_url):
        if not audio_url: return
        query = """
        MATCH (b:Bird {scientific_name: $sci})
        MERGE (a:Audio {bird_id: $sci})
        SET a.url = $url
        MERGE (b)-[:HAS_SOUND]->(a)
        """
        with self.driver.session() as session:
            session.run(query, sci=scientific_name, url=audio_url)

    def update_status(self, scientific_name, status_text):
        if not status_text: return
        query = """
        MATCH (b:Bird {scientific_name: $sci})
        MERGE (i:IUCN {bird_id: $sci})
        SET i.status = $status
        MERGE (b)-[:HAS_STATUS]->(i)
        """
        with self.driver.session() as session:
            session.run(query, sci=scientific_name, status=status_text)

    def update_ecology(self, scientific_name, data):
        if not data: return
        query = """
        MATCH (b:Bird {scientific_name: $sci})
        MERGE (e:Ecology {bird_id: $sci})
        SET e.diet = $diet, e.habitat = $habitat, e.migration = $mig
        MERGE (b)-[:HAS_ECOLOGY]->(e)
        """
        with self.driver.session() as session:
            session.run(query, sci=scientific_name, 
                        diet=data.get('diet'), 
                        habitat=data.get('habitat'), 
                        mig=data.get('migration'))

    def get_full_context(self, scientific_name):
        """Lấy toàn bộ dữ liệu (bao gồm cả các trường mới) để gửi cho LLM"""
        query = """
        MATCH (b:Bird {scientific_name: $sci})
        OPTIONAL MATCH (b)-[:HAS_INFO]->(w:WikiInfo)
        OPTIONAL MATCH (b)-[:HAS_SOUND]->(a:Audio)
        OPTIONAL MATCH (b)-[:HAS_STATUS]->(i:IUCN)
        OPTIONAL MATCH (b)-[:HAS_ECOLOGY]->(e:Ecology)
        
        RETURN b.common_name as Name,
               b.scientific_name as ScientificName,
               
               // --- CÁC TRƯỜNG MỚI ---
               b.image_url as ImageURL,
               b.map_url as MapURL,
               b.mass as Mass,
               b.wingspan as Wingspan,
               b.lifespan as Lifespan,
               b.food as MainFood,
               b.family as Family,
               b.conservation as Conservation_Wikidata,
               
               w.summary as Description,
               a.url as AudioURL,
               i.status as ConservationStatus,
               e.diet as Diet,
               e.habitat as Habitat
        """
        with self.driver.session() as session:
            rec = session.run(query, sci=scientific_name).single()
            if not rec:
                return "No data found in graph."
            return rec.data()