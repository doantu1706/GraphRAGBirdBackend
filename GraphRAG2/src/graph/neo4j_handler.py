import time
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import os
from src.config import Config

class Neo4jHandler:
    def __init__(self):
        # --- CẤU HÌNH DRIVER SIÊU BỀN (FIX LỖI KẾT NỐI NGẮT) ---
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            # 1. Tự động ngắt kết nối cũ sau 3 phút để tạo mới (tránh bị treo)
            max_connection_lifetime=180,
            # 2. Giữ kết nối sống
            keep_alive=True,
            # 3. Kiểm tra kết nối trước khi dùng (Tránh lỗi defunct connection)
            liveness_check_timeout=0, 
            # 4. Chờ tối đa 10s để lấy kết nối
            connection_acquisition_timeout=10.0
        )
        
        # Kiểm tra kết nối ngay khi khởi động
        try:
            self.driver.verify_connectivity()
            print("   ✅ [Neo4j] Connected to AuraDB successfully with Robust Config!")
        except Exception as e:
            print(f"   ❌ [Neo4j] Init Failed: {e}")
        
        print("   ⏳ Loading Embedding Model (Google Gemini)...")
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ MISSING GOOGLE_API_KEY. Please check .env or Environment Variables!")
            
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        self._init_indices()

    def _init_indices(self):
        try:
             with self.driver.session() as session:
                # Xóa index cũ nếu sai kích thước
                session.run("DROP INDEX bird_desc_index IF EXISTS")
        except:
            pass

        query = """
        CREATE VECTOR INDEX bird_desc_index IF NOT EXISTS
        FOR (w:WikiInfo) ON (w.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 768, 
            `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)

    def close(self):
        self.driver.close()

    def check_data_status(self, scientific_name):
        query = """
        MATCH (b:Bird {scientific_name: $sci})
        RETURN b,
               EXISTS((b)-[:HAS_INFO]->(:WikiInfo)) as has_wiki,
               EXISTS((b)-[:HAS_SOUND]->(:Audio)) as has_audio,
               EXISTS((b)-[:HAS_STATUS]->(:IUCN)) as has_status,
               EXISTS((b)-[:HAS_ECOLOGY]->(:Ecology)) as has_ecology,
               b.image_url IS NOT NULL as has_image,
               b.mass IS NOT NULL as has_mass
        """
        # Dùng session quản lý context để an toàn hơn
        with self.driver.session() as session:
            res = session.run(query, sci=scientific_name).single()
            if not res: return {"exists": False}
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

    def update_details(self, scientific_name, image_url=None, mass=None, map_url=None, wingspan=None, lifespan=None, food=None, family=None, conservation=None):
        if not any([image_url, mass, map_url, wingspan, lifespan, food, family, conservation]):
            return

        query = """
        MERGE (b:Bird {scientific_name: $sci})
        SET b.image_url = COALESCE($image, b.image_url),
            b.mass = COALESCE($mass, b.mass),
            b.map_url = COALESCE($map, b.map_url),
            b.wingspan = COALESCE($wingspan, b.wingspan),
            b.lifespan = COALESCE($lifespan, b.lifespan),
            b.food = COALESCE($food, b.food),
            b.family = COALESCE($family, b.family),
            b.conservation = COALESCE($conservation, b.conservation)
        """
        with self.driver.session() as session:
            session.run(query, sci=scientific_name, 
                        image=image_url, mass=mass, map=map_url, 
                        wingspan=wingspan, lifespan=lifespan, 
                        food=food, family=family, conservation=conservation)

    # --- HÀM XỬ LÝ LỖI 429 (BẠN ĐÃ LÀM ĐÚNG) ---
    def update_wiki(self, scientific_name, common_name, summary):
        if not summary: return
        
        vector = None
        for attempt in range(3):
            try:
                vector = self.embeddings.embed_query(summary)
                break 
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"      ⚠️ Google Rate Limit hit. Waiting 60s... (Attempt {attempt+1}/3)")
                    time.sleep(60) 
                else:
                    print(f"      ❌ Embedding Error: {e}")
                    return 

        if not vector:
            print("      ❌ Failed to embed summary after retries.")
            return

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
        query = """
        MATCH (b:Bird {scientific_name: $sci})
        OPTIONAL MATCH (b)-[:HAS_INFO]->(w:WikiInfo)
        OPTIONAL MATCH (b)-[:HAS_SOUND]->(a:Audio)
        OPTIONAL MATCH (b)-[:HAS_STATUS]->(i:IUCN)
        OPTIONAL MATCH (b)-[:HAS_ECOLOGY]->(e:Ecology)
        
        RETURN b.common_name as Name,
               b.scientific_name as ScientificName,
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
            if not rec: return "No data found in graph."
            return rec.data()
