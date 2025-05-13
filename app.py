import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import traceback
import uuid # Benzersiz ID'ler için

# -----------------------------------------------------------------------------
# SAYFA KONFİGÜRASYONU - İLK STREAMLIT KOMUTU OLMALI!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Çoklu Sohbet PDF Asistanı", page_icon="📚")
# -----------------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Streamlit Secrets ve OpenRouter Konfigürasyonu ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct:free")
LOCAL_EMBEDDING_MODEL_NAME = st.secrets.get("LOCAL_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

if not OPENROUTER_API_KEY:
    st.error("OpenRouter API anahtarı (LLM için) bulunamadı! Lütfen Streamlit Secrets bölümüne 'OPENROUTER_API_KEY' olarak ekleyin.")
    st.stop()

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Embedding modelini yükle ve cache'le
@st.cache_resource
def load_embeddings_model(model_name):
    print(f"Yerel embedding modeli yükleniyor: {model_name}")
    try:
        embeddings_instance = HuggingFaceEmbeddings(model_name=model_name)
        print("Yerel embedding modeli başarıyla yüklendi.")
        return embeddings_instance
    except Exception as e:
        st.error(f"Yerel embedding modeli ({model_name}) yüklenirken hata oluştu: {e}")
        st.error(traceback.format_exc())
        return None

embeddings_model_global = load_embeddings_model(LOCAL_EMBEDDING_MODEL_NAME)

if embeddings_model_global is None:
    st.stop()

# --- Yardımcı Fonksiyonlar ---
def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs:
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            except Exception as e:
                st.warning(f"'{pdf.name}' dosyasından metin çıkarılırken hata: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def create_vector_store_from_chunks(text_chunks, current_embeddings_model):
    if not text_chunks: return None
    if not current_embeddings_model: return None
    try:
        return FAISS.from_texts(texts=text_chunks, embedding=current_embeddings_model)
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata: {e}")
        st.error(traceback.format_exc())
        return None

def get_conversational_chain_prompt_template(): # Adını değiştirdim, sadece şablonu döndürsün diye
    prompt_template_str = """
    Sadece aşağıda verilen bağlamdaki bilgileri kullanarak soruyu yanıtlayın.
    Eğer cevap bağlamda yoksa, "Bilmiyorum, bu bilgi belgede bulunmuyor." deyin.
    Kesinlikle bağlam dışı bilgi kullanmayın veya cevap uydurmayın.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

# --- Session State Başlatma ---
if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # {session_id: session_data}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "prompt_template" not in st.session_state: # Prompt şablonunu bir kere oluşturup saklayalım
    st.session_state.prompt_template = get_conversational_chain_prompt_template()


# --- Oturum Yönetimi Fonksiyonları ---
def create_new_session():
    session_id = str(uuid.uuid4()) # Benzersiz ID
    session_name = f"Sohbet {len(st.session_state.sessions) + 1}"
    st.session_state.sessions[session_id] = {
        "id": session_id,
        "name": session_name,
        "pdf_names": [],
        "vector_store": None,
        "chat_history": [],
        "pdf_processed": False # Bu oturum için PDF'lerin işlenip işlenmediği
    }
    st.session_state.current_session_id = session_id
    return session_id

def get_active_session_data():
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.current_session_id]
    return None

def delete_session(session_id_to_delete):
    if session_id_to_delete in st.session_state.sessions:
        del st.session_state.sessions[session_id_to_delete]
        if st.session_state.current_session_id == session_id_to_delete:
            st.session_state.current_session_id = None # Aktif oturum silindiyse sıfırla
            # Veya listedeki ilk oturumu aktif yapabilirsiniz
            if st.session_state.sessions:
                st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]


# --- Streamlit Arayüzü ---
st.title("📚 Çoklu Sohbet PDF Asistanı")

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.header("Sohbet Oturumları")

    if st.button("➕ Yeni Sohbet Başlat", key="new_chat_button"):
        create_new_session()
        # Yeni sohbet başlatıldığında PDF yükleme alanını temizlemek için file_uploader'ı yeniden render et
        # Bu Streamlit'in bir bug'ı/özelliği, key değiştirerek yapılabilir ama şimdilik rerun yeterli olabilir.
        st.rerun()


    session_options = {sid: sdata["name"] for sid, sdata in st.session_state.sessions.items()}
    if not session_options and st.session_state.current_session_id is None: # Eğer hiç oturum yoksa ve aktif oturum da yoksa
        create_new_session() # Otomatik olarak ilk oturumu oluştur
        session_options = {sid: sdata["name"] for sid, sdata in st.session_state.sessions.items()}


    if session_options:
        # Selectbox'ın değeri olarak session_id kullanacağız
        selected_session_id = st.selectbox(
            "Aktif Sohbeti Seçin:",
            options=list(session_options.keys()),
            format_func=lambda sid: session_options[sid], # Kullanıcıya session_name'i göster
            index=list(session_options.keys()).index(st.session_state.current_session_id) if st.session_state.current_session_id in session_options else 0,
            key="session_selector"
        )
        if selected_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id
            st.rerun() # Seçim değişince sayfayı yenile

        active_session = get_active_session_data()

        if active_session:
            st.markdown("---")
            st.subheader(f"Aktif: {active_session['name']}")

            # PDF Yükleme ve İşleme (Aktif Oturum için)
            # PDF yükleme widget'ına dinamik bir key vererek, oturum değiştiğinde sıfırlanmasını sağlamaya çalışalım
            uploader_key = f"pdf_uploader_{active_session['id']}"
            uploaded_pdf_docs = st.file_uploader(
                "Bu sohbet için PDF dosyalarını yükleyin:",
                accept_multiple_files=True,
                type="pdf",
                key=uploader_key
            )

            if st.button("Seçili PDF'leri İşle", key=f"process_btn_{active_session['id']}"):
                if uploaded_pdf_docs:
                    with st.spinner("PDF'ler işleniyor..."):
                        active_session["pdf_names"] = [f.name for f in uploaded_pdf_docs]
                        raw_text = get_pdf_text(uploaded_pdf_docs)
                        if not raw_text.strip():
                            st.error("PDF'lerden metin çıkarılamadı.")
                            active_session["vector_store"] = None
                            active_session["pdf_processed"] = False
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            if not text_chunks:
                                st.error("Metin parçalara ayrılamadı.")
                                active_session["vector_store"] = None
                                active_session["pdf_processed"] = False
                            else:
                                vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                                if vector_store:
                                    active_session["vector_store"] = vector_store
                                    active_session["chat_history"] = [] # Yeni PDF'ler işlenince sohbeti sıfırla
                                    active_session["pdf_processed"] = True
                                    st.success(f"PDF(ler) '{active_session['name']}' için işlendi.")
                                    st.rerun() # Arayüzü güncellemek için
                                else:
                                    st.error("Vektör deposu oluşturulamadı.")
                                    active_session["pdf_processed"] = False
                else:
                    st.warning("Lütfen işlemek için PDF dosyası yükleyin.")
            
            if active_session["pdf_processed"] and active_session["pdf_names"]:
                 st.markdown("**İşlenmiş PDF(ler):**")
                 for name in active_session["pdf_names"]:
                     st.caption(f"- {name}")

            st.markdown("---")
            if st.button(f"'{active_session['name']}' Oturumunu Sil", type="secondary", key=f"delete_btn_{active_session['id']}"):
                session_id_to_delete = active_session['id']
                delete_session(session_id_to_delete)
                st.success(f"'{active_session['name']}' oturumu silindi.")
                st.rerun()
    else:
        st.sidebar.info("Henüz bir sohbet oturumu yok. Lütfen yeni bir tane başlatın.")


# --- Ana Sohbet Alanı ---
active_session_data = get_active_session_data()

if active_session_data:
    st.subheader(f"Sohbet: {active_session_data['name']}")

    # Sohbet geçmişini göster
    for message in active_session_data["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan girdi al
    if user_query := st.chat_input(f"'{active_session_data['name']}' hakkında sorun..."):
        if not active_session_data["vector_store"]:
            st.warning("Bu sohbet için henüz PDF işlenmedi veya vektör deposu oluşturulamadı. Lütfen kenar çubuğundan PDF yükleyip işleyin.")
        else:
            active_session_data["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response_text = ""
                try:
                    docs = active_session_data["vector_store"].similarity_search(query=user_query, k=4)
                    
                    if not docs:
                        full_response_text = "Belgede sorunuzla ilgili bir bilgi bulamadım."
                    else:
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        messages_for_llm = [
                            {"role": "system", "content": st.session_state.prompt_template.template.split("Soru:")[0].strip()},
                            {"role": "user", "content": f"Bağlam:\n{context_text}\n\nSoru: {user_query}\n\nCevap:"}
                        ]
                        
                        response_stream = llm_client.chat.completions.create(
                            model=LLM_MODEL_NAME, messages=messages_for_llm, stream=True,
                        )
                        
                        for chunk in response_stream:
                            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                                full_response_text += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response_text + "▌")
                    
                    message_placeholder.markdown(full_response_text)

                except Exception as e: # Daha genel hata yakalama
                    st.error(f"Yanıt alınırken bir hata oluştu: {e}")
                    st.error(traceback.format_exc())
                    full_response_text = "Üzgünüm, bir hata oluştu."
                    message_placeholder.markdown(full_response_text)

            active_session_data["chat_history"].append({"role": "assistant", "content": full_response_text})
else:
    st.info("Lütfen kenar çubuğundan bir sohbet seçin veya yeni bir tane başlatın.")

st.sidebar.markdown("---")
st.sidebar.caption(f"LLM: {LLM_MODEL_NAME}")
st.sidebar.caption(f"Embedding: {LOCAL_EMBEDDING_MODEL_NAME} (Yerel)")
