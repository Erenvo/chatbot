import streamlit as st
from openai import OpenAI # Google AI kullandığımız için bu aslında gerekmeyebilir, ama genel APIError için kalabilir.
                          # Veya google.api_core.exceptions gibi Google'a özel hataları yakalayabilirsiniz.
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # Google AI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser # Yapılandırılmış çıktı için
from pydantic import BaseModel, Field # Pydantic modelleri için
from typing import List, Optional, Union # Tip ipuçları için
import traceback
import uuid

# -----------------------------------------------------------------------------
# SAYFA KONFİGÜRASYONU - İLK STREAMLIT KOMUTU OLMALI!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Google AI PDF Asistanı", page_icon="✨📚")
# -----------------------------------------------------------------------------

# os.environ["TOKENIZERS_PARALLELISM"] = "false" # HuggingFace için gerekliydi

# --- Streamlit Secrets ve Google AI Konfigürasyonu ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_LLM_MODEL_NAME = st.secrets.get("GOOGLE_LLM_MODEL_NAME", "gemini-1.5-flash-latest")
GOOGLE_EMBEDDING_MODEL_NAME = st.secrets.get("GOOGLE_EMBEDDING_MODEL_NAME", "models/embedding-001")

if not GOOGLE_API_KEY:
    st.error("Google API anahtarı bulunamadı! Lütfen Streamlit Secrets bölümüne 'GOOGLE_API_KEY' olarak ekleyin.")
    st.stop()

# --- Pydantic Modelleri Tanımlama (Yapılandırılmış Çıktı İçin) ---
class GeneratedQuestionPydantic(BaseModel):
    question: str = Field(description="PDF içeriği hakkında anlamlı ve spesifik bir soru.")

class DocumentAnalysisOutput(BaseModel):
    summary: Optional[str] = Field(default=None, description="PDF içeriğinin kısa bir özeti (3-4 cümle).")
    questions: Optional[List[GeneratedQuestionPydantic]] = Field(default=None, description="PDF içeriğine dayalı olarak türetilmiş 2 ila 3 adet spesifik soru.")

# --- Langchain Output Parser ---
output_parser_global = PydanticOutputParser(pydantic_object=DocumentAnalysisOutput)

# --- LLM ve Embedding İstemcileri ---
@st.cache_resource # Embedding modelini global olarak cache'le
def load_google_embedding_model(api_key, model_name):
    try:
        print(f"Google AI Embedding istemcisi yükleniyor: {model_name}")
        model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
        print("Google AI Embedding istemcisi başarıyla yüklendi.")
        return model
    except Exception as e:
        st.error(f"Google AI Embedding modeli ({model_name}) yüklenirken hata: {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_resource # LLM client'ını da cache'leyebiliriz
def load_google_llm_client(api_key, model_name):
    try:
        print(f"Google AI LLM istemcisi yükleniyor: {model_name}")
        client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2, # Biraz daha deterministik olması için
            # convert_system_message_to_human=True # Gerekirse
        )
        print("Google AI LLM istemcisi başarıyla yüklendi.")
        return client
    except Exception as e:
        st.error(f"Google AI LLM istemcisi ({model_name}) oluşturulurken hata: {e}")
        st.error(traceback.format_exc())
        return None

embeddings_model_global = load_google_embedding_model(GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL_NAME)
llm_client_global = load_google_llm_client(GOOGLE_API_KEY, GOOGLE_LLM_MODEL_NAME)

if embeddings_model_global is None or llm_client_global is None:
    st.error("LLM veya Embedding modeli yüklenemedi. Lütfen API anahtarınızı ve model adlarınızı kontrol edin.")
    st.stop()

# --- Yardımcı Fonksiyonlar ---
def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs:
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text();
                    if page_text: text += page_text
            except Exception as e: st.warning(f"'{pdf.name}' dosyasından metin çıkarılırken hata: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def create_vector_store_from_chunks(text_chunks, current_embeddings_model):
    if not text_chunks or not current_embeddings_model: return None
    try: return FAISS.from_texts(texts=text_chunks, embedding=current_embeddings_model)
    except Exception as e: st.error(f"Vektör deposu oluşturulurken hata: {e}"); st.error(traceback.format_exc()); return None

# --- Prompt Şablonları ---
@st.cache_data # Prompt template'leri de cache'leyebiliriz, çünkü değişmiyorlar
def get_rag_prompt_template_cached():
    prompt_template_str = """
    SENİN GÖREVİN: Sadece ve sadece aşağıda "Bağlam:" olarak verilen metindeki bilgileri kullanarak "Soru:" kısmındaki soruyu yanıtlamaktır.
    KESİNLİKLE DIŞARIDAN BİLGİ KULLANMA, YORUM YAPMA, EK AÇIKLAMA EKLEME VEYA CEVAP UYDURMA.
    Cevabın SADECE ve SADECE "Bağlam:" içindeki bilgilere dayanmalıdır.

    Eğer "Soru:" kısmındaki soruya cevap "Bağlam:" içinde bulunmuyorsa, şu cevabı ver:
    "Bu bilgi sağlanan belgede bulunmuyor."
    BU CEVABIN DIŞINDA HİÇBİR ŞEY EKLEME. Örneğin, "Bu bilgi belgede yok ama genel olarak şöyledir..." GİBİ BİR AÇIKLAMA YAPMA.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

@st.cache_data
def get_structured_analysis_prompt_template_cached(_parser: PydanticOutputParser): # Parser argüman olarak alınmalı
    prompt_template_str = """
    SENİN GÖREVİN: Aşağıda "Bağlam:" olarak verilen metni analiz etmektir.
    Kullanıcı bu belge hakkında genel bir fikir edinmek istiyor.

    LÜTFEN AŞAĞIDAKİLERDEN BİRİNİ YAP (SADECE BİRİNİ SEÇ VE ONA UYGUN FORMATTA ÇIKTI VER):
    1. Metnin ana temalarını ve içeriğini özetleyen kısa (3-4 CÜMLE) bir açıklama üret. (Bu durumda 'summary' alanını doldur.)
    2. VEYA, bu metne dayanarak kullanıcının sorabileceği 2 İLA 3 ADET anlamlı ve spesifik soru türet. Bu sorular, metnin farklı önemli kısımlarını kapsamalıdır. (Bu durumda 'questions' alanını doldur.)

    Kesinlikle dışarıdan bilgi kullanma. Sadece sağlanan bağlamı kullan.

    {format_instructions}

    Bağlam:
    {context}

    İstenen Çıktı (Yukarıdaki format talimatlarına uygun JSON):"""
    return PromptTemplate(
        template=prompt_template_str,
        input_variables=["context"],
        partial_variables={"format_instructions": _parser.get_format_instructions()}
    )

# --- Session State Başlatma ---
if "sessions" not in st.session_state: st.session_state.sessions = {}
if "current_session_id" not in st.session_state: st.session_state.current_session_id = None
# Prompt template'leri session state'e yüklemeye gerek yok, cache'li fonksiyonlardan çağırabiliriz.

# --- Oturum Yönetimi Fonksiyonları ---
def create_new_session():
    session_id = str(uuid.uuid4()); session_name = f"Sohbet {len(st.session_state.sessions) + 1}"
    st.session_state.sessions[session_id] = {
        "id": session_id, "name": session_name, "pdf_names": [], "all_text_chunks": [],
        "vector_store": None, "chat_history": [], "pdf_processed": False
    }
    st.session_state.current_session_id = session_id; return session_id

def get_active_session_data():
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.current_session_id]
    return None

def delete_session(session_id_to_delete):
    if session_id_to_delete in st.session_state.sessions:
        del st.session_state.sessions[session_id_to_delete]
        if st.session_state.current_session_id == session_id_to_delete:
            st.session_state.current_session_id = None
            if st.session_state.sessions: st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]

# --- Streamlit Arayüzü ---
st.title("✨ Google AI Destekli PDF Asistanı 📚")

with st.sidebar:
    st.header("Sohbet Oturumları")
    if st.button("➕ Yeni Sohbet Başlat", key="new_chat_button", use_container_width=True):
        create_new_session()
        st.rerun()

    session_options = {sid: sdata["name"] for sid, sdata in st.session_state.sessions.items()}
    if not session_options and st.session_state.current_session_id is None:
        create_new_session() # İlk oturumu oluştur
        session_options = {sid: sdata["name"] for sid, sdata in st.session_state.sessions.items()} # Seçenekleri güncelle

    if session_options: # Sadece oturum varsa göster
        selected_session_id = st.selectbox(
            "Aktif Sohbeti Seçin:", options=list(session_options.keys()),
            format_func=lambda sid: session_options.get(sid, "Bilinmeyen Oturum"), # .get() ile daha güvenli
            index=list(session_options.keys()).index(st.session_state.current_session_id) if st.session_state.current_session_id in session_options else 0,
            key="session_selector"
        )
        if selected_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id
            st.rerun()

        active_session = get_active_session_data()
        if active_session:
            st.markdown("---"); st.subheader(f"Aktif: {active_session['name']}")
            uploader_key = f"pdf_uploader_{active_session['id']}" # Her oturum için farklı key
            uploaded_pdf_docs = st.file_uploader(
                "Bu sohbet için PDF dosyalarını yükleyin:",
                accept_multiple_files=True, type="pdf", key=uploader_key
            )
            if st.button("Seçili PDF'leri İşle", key=f"process_btn_{active_session['id']}", use_container_width=True):
                if uploaded_pdf_docs:
                    with st.spinner(f"'{active_session['name']}' için PDF'ler işleniyor..."):
                        active_session["pdf_names"] = [f.name for f in uploaded_pdf_docs]
                        raw_text = get_pdf_text(uploaded_pdf_docs)
                        if not raw_text.strip():
                            st.error("PDF'lerden metin çıkarılamadı."); active_session["all_text_chunks"] = []
                            active_session["vector_store"] = None; active_session["pdf_processed"] = False
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            active_session["all_text_chunks"] = text_chunks # Genel sorular için sakla
                            if not text_chunks:
                                st.error("Metin parçalara ayrılamadı.");
                                active_session["vector_store"] = None; active_session["pdf_processed"] = False
                            else:
                                vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                                if vector_store:
                                    active_session["vector_store"] = vector_store
                                    active_session["chat_history"] = [] # Yeni PDF'lerde sohbeti sıfırla
                                    active_session["pdf_processed"] = True
                                    st.success(f"PDF(ler) '{active_session['name']}' için başarıyla işlendi.")
                                    st.rerun() # Arayüzü ve sohbet alanını güncelle
                                else:
                                    st.error("Vektör deposu oluşturulamadı."); active_session["pdf_processed"] = False
                else:
                    st.warning("Lütfen işlemek için en az bir PDF dosyası yükleyin.")
            
            if active_session.get("pdf_processed") and active_session.get("pdf_names"):
                 st.markdown("**İşlenmiş PDF(ler):**")
                 for name in active_session["pdf_names"]: st.caption(f"- {name}")

            st.markdown("---")
            if st.button(f"🗑️ '{active_session['name']}' Oturumunu Sil", type="secondary", key=f"delete_btn_{active_session['id']}", use_container_width=True):
                session_name_deleted = active_session['name'] # Silmeden önce ismi al
                delete_session(active_session['id'])
                st.success(f"'{session_name_deleted}' oturumu silindi."); st.rerun()
    else:
        st.sidebar.info("Henüz bir sohbet oturumu yok. Lütfen yeni bir tane başlatın.")

# Ana Sohbet Alanı
active_session_data = get_active_session_data()

if active_session_data:
    st.header(f"Sohbet: {active_session_data['name']}")
    # Sohbet geçmişini göster
    for message in active_session_data["chat_history"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    # Kullanıcıdan girdi al
    if user_query := st.chat_input(f"'{active_session_data['name']}' oturumundaki PDF(ler) hakkında sorun..."):
        # Önce PDF'lerin işlenip işlenmediğini kontrol et
        pdf_is_ready = active_session_data.get("pdf_processed", False) and active_session_data.get("vector_store") is not None
        general_query_ready = active_session_data.get("pdf_processed", False) and active_session_data.get("all_text_chunks")
        
        general_query_keywords = ["ne anlatıyor", "konusu ne", "özetle", "ne bulunur", "neler var", "bahset", "içeriği", "genel bakış"]
        is_general_query = any(keyword in user_query.lower() for keyword in general_query_keywords)

        if not (pdf_is_ready or (is_general_query and general_query_ready)):
            st.warning("Lütfen önce bu sohbet için PDF yükleyip işleyin.")
        else:
            active_session_data["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.markdown(user_query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty(); full_response_text = ""
                try:
                    if is_general_query and general_query_ready:
                        st.write("DEBUG: Genel soru algılandı, yapılandırılmış analiz yapılıyor...")
                        # İlk N chunk'ı veya belirli bir token limitine kadar olanı al
                        # Bu, modelin context window'una göre ayarlanmalı.
                        # Gemini 1.5 Flash büyük bir context window'a sahip.
                        context_chunks = active_session_data["all_text_chunks"][:20] # Örnek: İlk 20 chunk
                        context_text_for_analysis = "\n\n".join(context_chunks)
                        
                        analysis_prompt_template = get_structured_analysis_prompt_template_cached(output_parser_global)
                        formatted_analysis_prompt = analysis_prompt_template.format(context=context_text_for_analysis)
                        
                        raw_llm_output = ""
                        for chunk in llm_client_global.stream(formatted_analysis_prompt):
                            if hasattr(chunk, 'content'):
                                raw_llm_output += chunk.content
                                message_placeholder.markdown(raw_llm_output + "▌")
                        message_placeholder.markdown(raw_llm_output) # Son ham çıktı

                        try:
                            # JSON ayıklama (LLM bazen ```json ... ``` bloğu ekler)
                            json_part = raw_llm_output
                            if "```json" in raw_llm_output:
                                json_part = raw_llm_output.split("```json", 1)[1].split("```", 1)[0].strip()
                            elif "```" in raw_llm_output and raw_llm_output.strip().startswith("```") and raw_llm_output.strip().endswith("```"):
                                json_part = raw_llm_output.strip()[3:-3].strip()
                            
                            parsed_output: DocumentAnalysisOutput = output_parser_global.parse(json_part)
                            
                            if parsed_output.summary:
                                full_response_text = f"**Belge Özeti:**\n{parsed_output.summary}"
                            elif parsed_output.questions:
                                questions_md = "**Bu belge hakkında sorabileceğiniz bazı sorular:**\n"
                                for i, q_obj in enumerate(parsed_output.questions):
                                    questions_md += f"{i+1}. {q_obj.question}\n"
                                full_response_text = questions_md
                            else: # İkisinden biri olmalı, prompt'a göre. Ama fallback.
                                full_response_text = "Belge hakkında genel bir özet veya soru üretemedim, ancak LLM'den gelen yanıt yukarıdadır."
                        except Exception as parse_error:
                            st.error(f"LLM çıktısı (analiz için) parse edilirken hata: {parse_error}")
                            st.caption("LLM'den gelen ham çıktı (parse edilemedi):")
                            st.code(raw_llm_output, language='text')
                            full_response_text = "Üzgünüm, belge hakkında genel bir bilgi veya soru üretemedim. Ham yanıt yukarıda."
                        message_placeholder.markdown(full_response_text)

                    elif pdf_is_ready: # Spesifik soru (RAG)
                        docs = active_session_data["vector_store"].similarity_search(query=user_query, k=4)
                        if not docs:
                            full_response_text = "Bu bilgi sağlanan belgede bulunmuyor."
                        else:
                            context_text_for_rag = "\n\n".join([doc.page_content for doc in docs])
                            rag_prompt_template = get_rag_prompt_template_cached()
                            formatted_rag_prompt = rag_prompt_template.format(context=context_text_for_rag, question=user_query)
                            
                            for chunk in llm_client_global.stream(formatted_rag_prompt):
                                if hasattr(chunk, 'content'):
                                    full_response_text += chunk.content
                                    message_placeholder.markdown(full_response_text + "▌")
                        message_placeholder.markdown(full_response_text)
                    else: # Bu duruma düşmemeli ama fallback
                        full_response_text = "Lütfen önce PDF yükleyip işleyin."
                        message_placeholder.markdown(full_response_text)

                except Exception as e:
                    st.error(f"Yanıt alınırken beklenmedik bir hata oluştu: {e}")
                    st.error(traceback.format_exc()); full_response_text = "Üzgünüm, bir hata oluştu."
                    message_placeholder.markdown(full_response_text)
            active_session_data["chat_history"].append({"role": "assistant", "content": full_response_text})
else:
    st.info("Başlamak için kenar çubuğundan bir sohbet seçin veya 'Yeni Sohbet Başlat'a tıklayın.")

st.sidebar.markdown("---")
st.sidebar.caption(f"LLM: {GOOGLE_LLM_MODEL_NAME} (Google AI)")
st.sidebar.caption(f"Embedding: {GOOGLE_EMBEDDING_MODEL_NAME} (Google AI)")
