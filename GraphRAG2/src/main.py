import os
import sys
import json
from typing import Dict, List, Any

# Import LangChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import Config
from src.config import Config

# Import Database Handler
from src.graph.neo4j_handler import Neo4jHandler

# Import Data Loaders
from src.data_loaders.wikidata import WikidataFetcher
from src.data_loaders.wikipedia import WikipediaFetcher
from src.data_loaders.xenocanto import XenoCantoFetcher
from src.data_loaders.iucn import IUCNFetcher
from src.data_loaders.birdspedia import BirdspediaFetcher

class BirdGraphRAG:
    def __init__(self):
        print("🚀 Initializing BirdGraphRAG System...")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.graph = Neo4jHandler()
        self.wikidata = WikidataFetcher()
        self.wiki = WikipediaFetcher()
        self.xenocanto = XenoCantoFetcher()
        self.iucn = IUCNFetcher()
        self.birdspedia = BirdspediaFetcher()
        
        # --- QUẢN LÝ ĐA PHIÊN (MULTI-SESSION) ---
        # Thay vì self.chat_history = [], ta dùng dict để lưu theo session_id
        # Cấu trúc: { "session_1": [msg1, msg2], "session_2": [] }
        self.sessions = {} 
        
        print("✅ System Ready!\n")

    def get_session_history(self, session_id: str):
        """Lấy lịch sử của một phiên cụ thể"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def reset_history(self, session_id: str = "default"):
        """Xóa lịch sử của một phiên"""
        self.sessions[session_id] = []
        print(f"🧹 Chat history for session '{session_id}' has been reset!")

    def _contextualize_query(self, raw_query: str, session_id: str) -> str:
        """Viết lại câu hỏi dựa trên lịch sử của phiên hiện tại"""
        history = self.get_session_history(session_id)
        if not history:
            return raw_query

        # Lấy 4 tin nhắn gần nhất
        history_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in history[-4:]])

        prompt = PromptTemplate.from_template("""
        Task: Rewrite the 'Latest Question' into a standalone question based on 'Chat History'.
        - If the user uses pronouns (it, she, he, that bird), replace them with the specific bird name from history.
        - If the 'Latest Question' is unrelated to the history (e.g., a new topic), leave it as is.
        
        Chat History:
        {history}
        
        Latest Question: {question}
        
        Standalone Question:
        """)
        
        try:
            chain = prompt | self.llm
            rewritten = chain.invoke({"history": history_str, "question": raw_query}).content.strip()
            if rewritten != raw_query:
                print(f"🔄 [Context] '{raw_query}' -> '{rewritten}'")
            return rewritten
        except:
            return raw_query

    def _analyze_intent_and_entity(self, query: str) -> Dict:
        """BỘ LỌC THÔNG MINH + PHÂN LOẠI"""
        prompt = f"""
        Analyze this query: "{query}"
        
        STRICTLY Determine if the query is related to Ornithology (Birds).
        
        OUTPUT JSON ONLY with keys:
        1. "is_relevant": boolean.
           - TRUE if it asks about birds, lists of birds, bird stats, or biology.
           - FALSE if it asks about Sports, Cooking, Coding, Politics, or Gibberish.
        
        2. "intent": string.
           - "lookup": Ask about a specific bird (e.g., "What is a Kingfisher?").
           - "filter_list": Ask for a list/filtering (e.g., "Birds in Vietnam", "Blue birds", "Birds < 10g").
        
        3. "bird_name": string or null.
           - If intent is "lookup", extract common name. Else null.
           
        4. "lookup_type": string.
           - "general": "Tell me about X", "What is X".
           - "specific": "Color of X", "Weight of X", "Map of X".
        """
        
        try:
            res = self.llm.invoke(prompt).content.strip()
            clean_res = res.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_res)
        except Exception as e:
            print(f"⚠️ Intent parsing error: {e}")
            return {"is_relevant": False, "intent": "lookup", "bird_name": None}

    def _generate_sparql_query(self, user_query: str) -> str:
        """
        Dạy LLM viết code SPARQL chuẩn xác, CÓ XỬ LÝ ĐƠN VỊ.
        """
        prompt = f"""
        You are a SPARQL Expert for Wikidata.
        Task: Convert User Query into a valid SPARQL query to find birds.
        
        --- CRITICAL SCHEMA ---
        1. MUST BE A BIRD: ?bird wdt:P171+ wd:Q5113.
        2. Instance of Taxon: ?bird wdt:P31 wd:Q16521.
        3. Image: ?bird wdt:P18 ?image.
        
        --- HANDLING MASS (Weight) ---
        Property: p:P2067 (Statement), psn:P2067 (Normalized Value).
        IMPORTANT: Wikidata stores mass in different units (kg, g). 
        To filter correctly, use the 'Normalized Value' node which usually defaults to Grams or Kilograms, OR specifically convert units.
        
        BETTER STRATEGY: Use `psn:P2067` (Normalized quantity) inside a path.
        Example Path: ?bird p:P2067/psn:P2067 ?massNode. ?massNode wikibase:quantityAmount ?mass. ?massNode wikibase:quantityUnit ?unit.
        
        HOWEVER, for simplicity and speed with LLMs, use this TRICK:
        Use `?bird p:P2067 ?statement. ?statement ps:P2067 ?massVal. ?statement psv:P2067 ?massNode. ?massNode wikibase:quantityAmount ?mass. ?massNode wikibase:quantityUnit ?unit.`
        
        Then FILTER based on unit:
        - If unit is Gram (wd:Q41803), keep ?mass.
        - If unit is Kilogram (wd:Q11570), ?mass * 1000.
        
        **SIMPLIFIED PROMPT FOR YOU:**
        Just use the raw value but assume mixed units might exist. 
        Actually, the best way for a "List birds under 20g" query is:
        
        SELECT DISTINCT ?birdLabel ?image ?mass WHERE {{
          ?bird wdt:P31 wd:Q16521; wdt:P171+ wd:Q5113.
          
          # Get Mass and Unit
          ?bird p:P2067 ?statement.
          ?statement ps:P2067 ?massVal.
          ?statement psv:P2067 ?valNode.
          ?valNode wikibase:quantityAmount ?amount.
          ?valNode wikibase:quantityUnit ?unit.
          
          # Normalize to Grams
          BIND(IF(?unit = wd:Q11570, ?amount * 1000, ?amount) AS ?mass).
          
          # Filter (User asked for < 20g)
          FILTER(?mass < 20).
          
          ?bird wdt:P18 ?image.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "vi,en". }}
        }} ORDER BY ASC(?mass) LIMIT 15
        
        --- RULES ---
        1. Always SELECT ?birdLabel ?image ?mass.
        2. IF querying Mass/Weight: YOU MUST NORMALIZE UNITS (kg -> g) using BIND/IF as shown above.
        3. Limit 5.
        4. Output ONLY raw SPARQL.
        
        User: "{user_query}"
        Output:
        """
        try:
            return self.llm.invoke(prompt).content.strip().replace("```sparql", "").replace("```", "")
        except:
            return ""

    def _lazy_load_data(self, scientific_name: str, common_name: str, status: Dict):
        # 0. DỮ LIỆU WIKIDATA (Full)
        if not status.get('has_image') or not status.get('has_mass'):
            print(f"   📥 [Fetch] Wikidata Details for '{common_name}'...")
            wiki_data = self.wikidata.get_bird_data(common_name)
            if wiki_data:
                self.graph.update_details(
                    scientific_name, 
                    image_url=wiki_data.get('image_url'), 
                    mass=wiki_data.get('mass'),
                    map_url=wiki_data.get('map_url'),
                    wingspan=wiki_data.get('wingspan'),
                    lifespan=wiki_data.get('lifespan'),
                    food=wiki_data.get('food'),
                    family=wiki_data.get('family'),
                    conservation=wiki_data.get('conservation')
                )

        # 1. Wiki Description
        if not status.get('has_wiki'):
            print(f"   📥 [Fetch] Wikipedia for '{common_name}'...")
            summary = self.wiki.get_summary(common_name, lang='vi')
            if summary:
                self.graph.update_wiki(scientific_name, common_name, summary)
        
        # 2. Audio
        if not status.get('has_audio'):
            print(f"   📥 [Fetch] Audio for '{scientific_name}'...")
            audio_data = self.xenocanto.get_audio(scientific_name)
            if audio_data:
                self.graph.update_audio(scientific_name, audio_data['url'])

        # 3. Ecology
        if not status.get('has_ecology'):
            eco_data = self.birdspedia.fetch_ecology_data(scientific_name)
            if eco_data:
                self.graph.update_ecology(scientific_name, eco_data)

    def process_turn(self, user_input: str, session_id: str = "default") -> str:
        print(f"👤 User ({session_id}): {user_input}")
        
        # Lấy lịch sử của đúng phiên này
        history = self.get_session_history(session_id)
        
        # 1. Ngữ cảnh
        standalone_query = self._contextualize_query(user_input, session_id)

        # 2. Phân tích (BỘ LỌC & Ý ĐỊNH)
        analysis = self._analyze_intent_and_entity(standalone_query)
        
        # [GUARDRAIL] Chặn câu hỏi không liên quan
        if not analysis.get('is_relevant'):
            print("   ⛔ Blocked: Irrelevant query.")
            refusal = "Xin lỗi, tôi chỉ trả lời các câu hỏi về **loài chim**. Vui lòng đặt câu hỏi đúng chủ đề! 🐦"
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=refusal))
            return refusal

        intent = analysis.get('intent', 'lookup')
        lookup_type = analysis.get('lookup_type', 'general')

        # --- CASE 1: XỬ LÝ DANH SÁCH / LỌC (TEXT-TO-SPARQL) ---
        if intent == "filter_list":
            print("   🤖 Generative SPARQL Mode activated.")
            sparql_query = self._generate_sparql_query(standalone_query)
            
            if hasattr(self.wikidata, 'execute_generated_sparql'):
                results = self.wikidata.execute_generated_sparql(sparql_query)
                if results:
                    msg = f"Dưới đây là danh sách các loài chim tôi tìm thấy dựa trên yêu cầu của bạn:\n"
                    for r in results:
                        msg += f"- **{r['name']}** {r.get('info', '')}\n![{r['name']}]({r['image']})\n"
                    
                    history.append(HumanMessage(content=user_input))
                    history.append(AIMessage(content=msg))
                    return msg
            
            return "Xin lỗi, tôi không tìm thấy danh sách phù hợp trên hệ thống dữ liệu."

        # --- CASE 2: TRA CỨU CỤ THỂ (LOOKUP) ---
        bird_name = analysis.get('bird_name')
        
        if not bird_name:
            response = self.llm.invoke(f"Answer briefly about birds: {standalone_query}").content
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=response))
            return response

        print(f"   🐦 Target Bird: {bird_name}")
        
        # Định danh
        bird_data = self.wikidata.get_bird_data(bird_name)
        if bird_data:
            sci_name = bird_data['scientific_name']
        else:
            sci_name = self.llm.invoke(f"Scientific name of '{bird_name}'? Return only name.").content.strip()
        print(f"   🔬 Scientific Name: {sci_name}")

        # Kiểm tra & Tải dữ liệu
        status = self.graph.check_data_status(sci_name)
        if not status['exists']:
            print("   ✨ New Entity detected!")
        self._lazy_load_data(sci_name, bird_name, status)

        # Lấy Context
        context_data = self.graph.get_full_context(sci_name)
        
        # Tạo Prompt trả lời
        if intent == "specific" or lookup_type == "specific":
            # Chế độ trả lời ngắn gọn (Sửa lỗi hiển thị Map)
            system_instructions = """
            Answer DIRECTLY and CONCISELY based on the user's specific question.
            - Answer ONLY what is asked (e.g., color, food).
            - NO summaries or long intros.
            
            CRITICAL FOR IMAGES & MAPS:
            - If asking for "map", "distribution", "nơi sống": 
              CHECK 'MapURL'. IF exists, DISPLAY AS IMAGE: ![Bản đồ phân bố](MapURL). 
            - If asking for appearance:
              Include bird image: ![Bird Image](ImageURL).
            """
        else:
            # Chế độ trả lời đầy đủ
            system_instructions = """
            Provide a COMPREHENSIVE Guide.
            
            1. VISUALS (MANDATORY): 
               - Start with the main image: ![Bird Image](ImageURL).
               - IF 'MapURL' is available, DISPLAY IT below the main image or in Habitat section: ![Bản đồ phân bố](MapURL).
               
            2. SUMMARY: Provide a 'Thông tin nhanh' list (Name, Mass, Wingspan, Conservation).
            3. DETAILS: Describe appearance, habitat, and behavior.
            4. AUDIO: End with [🔊 Nghe giọng hót](AudioURL).
            """

        rag_prompt = f"""
        You are an expert Ornithologist. Answer in VIETNAMESE.
        
        --- INSTRUCTIONS ---
        {system_instructions}
        
        --- CONTEXT ---
        {context_data}
        
        --- QUESTION ---
        {standalone_query}
        """
        
        final_response = self.llm.invoke(rag_prompt).content
        
        # Cập nhật lịch sử (Đúng phiên)
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=final_response))
        
        return final_response

    def close(self):
        self.graph.close()
        print("👋 Connection closed.")

if __name__ == "__main__":
    agent = BirdGraphRAG()
    try:
        while True:
            user_input = input("\n👉 Bạn: ")
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip(): continue
            try:
                # Test trên console dùng session mặc định
                print(f"\n🤖 Bot: {agent.process_turn(user_input, 'console_session')}")
            except Exception as e:
                print(f"❌ Error: {e}")
    finally:
        agent.close()