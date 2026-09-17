"""
src/rag.py

Sigue la lógica del sistema RAG y la integración con el modelo predictivo XGBoost, extraída de 04_RAG_pipeline.ipynb para ser reutilizada por la API (src/main.py).

No recrea ni reindexa la colección de Qdrant porque se asume que ya fue creada e indexada por el notebook en las secciones 2-4. Si la colección no existe, lanza un 
error claro en vez de fallar silenciosamente con resultados vacíos.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Optional, TypedDict
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import convert_to_openai_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient


# ------------------------------------------------------------ Rutas --------------------------------------------------------------- 

# Raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Modelos
MODELS_DIR = BASE_DIR / "models"

# Variables de entorno
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "No se encontró OPENAI_API_KEY en el archivo .env"

# Configuración del rag
with open(MODELS_DIR / "config_rag.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# -------------------------------------------------------- Modelos y Clientes ---------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=CONFIG["embedding_model"],
    model_kwargs={'device': 'cpu'}
)
 
llm = ChatOpenAI(model=CONFIG["llm_model"], api_key=OPENAI_API_KEY, temperature=0)
 
QDRANT_HOST = CONFIG["qdrant_host"]
QDRANT_PORT = CONFIG["qdrant_port"]
 
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
 
COLLECTION_NAME = CONFIG["qdrant_collection"]
 
if not qdrant_client.collection_exists(COLLECTION_NAME):
    raise RuntimeError(
        f"La colección '{COLLECTION_NAME}' no existe en Qdrant. "
        "Ejecuta 04_RAG_pipeline.ipynb (Secciones 2-4) antes de arrancar la API."
    )
 
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
 
with open(MODELS_DIR / "xgboost_final.pkl", "rb") as f:
    modelo_xgb = pickle.load(f)
 
with open(MODELS_DIR / "esquema_categorias.json", "r", encoding="utf-8") as f:
    esquema = json.load(f)
 
# Umbral de decisión para β=1.5, como se decidió en el 03_modelo_ML.ipynb
UMBRAL_DECISION = CONFIG["umbral_decision"]


# -------------------------------- Integración del modelo predictivo (Como la Sección 6 de 04_RAG_pipeline.ipynb) ------------------------------------------

def construir_paciente(datos: dict) -> pd.DataFrame:
    fila = {}
    for col, categorias in esquema['categoricas'].items():
        valor = datos.get(col)
        if valor is not None and valor not in categorias:
            raise ValueError(f"Valor '{valor}' no es válido para '{col}'. Opciones: {categorias}")
        fila[col] = pd.Categorical([valor], categories=categorias)
 
    for col, rango in esquema['numericas'].items():
        valor = datos.get(col)
        fila[col] = [valor]
        if valor is not None:
            if valor < rango['min']:
                print(f" AVISO!!!! {col}={valor} Está por debajo del rango de entrenamiento "
                      f"(mínimo visto: {rango['min']}) — revisa si es un error de escritura.")
            elif valor > rango['max']:
                print(f"OJO!!! {col}={valor} Es posterior al dato más reciente usado para "
                      f"entrenar el modelo ({rango['max']}). Es el caso de uso esperado para "
                      f"pacientes actuales, pero la predicción extrapola la tendencia aprendida "
                      f"hasta {rango['max']} sin conocer cambios y avances en los tratamientos.")
 
    df_paciente = pd.DataFrame(fila)
    orden_entrenamiento = list(esquema['numericas'].keys()) + list(esquema['categoricas'].keys())
 
    return df_paciente[orden_entrenamiento]
 
 
def predecir_paciente(datos: dict) -> dict:
    """Devuelve la predicción del modelo para un paciente, con contexto para el RAG."""
    X_paciente = construir_paciente(datos)
    proba = modelo_xgb.predict_proba(X_paciente)[0, 1]
    prediccion = int(proba >= UMBRAL_DECISION)
 
    return {
        "probabilidad_no_supervivencia_5a": round(float(proba), 3),
        "prediccion_riesgo_alto": bool(prediccion),
        "umbral_usado": UMBRAL_DECISION,
        "datos_paciente": datos,
    }


# -------------------------------------------- Glosario de variables y contexto (Sección 7) --------------------------------------------------

GLOSARIO_VARIABLES = """
Glosario de las variables usadas por el modelo predictivo (para tu interpretación, no lo repitas
literalmente al usuario salvo que pregunte específicamente por el significado de una variable):

- age_group: grupo de edad del paciente al diagnóstico.
- sex: sexo del paciente.
- tumor_type: subtipo histológico del tumor (Ewing sarcoma u Osteosarcoma).
- primary_site: localización anatómica donde se originó el tumor.
- stage: estadio de extensión del tumor (Localized/Regional/Distant). Si aparece como ausente
  o no especificado, significa que el registro histórico no recogió ese dato (por ejemplo, por
  el año de diagnóstico del paciente) — NO significa que el diagnóstico del tumor sea incierto
  o esté sin confirmar. El diagnóstico del tipo de tumor es independiente de si se registró su
  estadio.
- surgery_code: tipo de cirugía realizada (o si no se realizó cirugía).
- radiation: si el paciente recibió radioterapia.
- chemotherapy: si el paciente recibió quimioterapia.
- year_diagnosis: año en que se diagnosticó el tumor.

Estas variables proceden de un registro poblacional (SEER), no de una historia clínica
completa: no incluyen biopsias, pruebas de imagen, marcadores moleculares ni comorbilidades.
"""

def formatear_contexto_prediccion(prediccion: dict, perfil: str) -> str:
    if not prediccion:
        return ""
    p = prediccion

    # Perfil del oncólogo/médico
    if perfil == "oncologo":
        return f"""
        PREDICCIÓN DEL MODELO PARA ESTE PACIENTE CONCRETO (dato obligatorio a mencionar en tu
        respuesta si la pregunta trata sobre pronóstico, riesgo o supervivencia):
        - Probabilidad estimada de no supervivencia a 5 años: {p['probabilidad_no_supervivencia_5a']*100:.1f}%
        - Clasificación de riesgo: {'Alto' if p['prediccion_riesgo_alto'] else 'No alto'} (umbral de decisión: {p['umbral_usado']})
        - Datos del paciente usados: {p['datos_paciente']}
        Es una estimación de apoyo a la decisión clínica basada en datos históricos de registro
        (SEER), no un diagnóstico ni una certeza.
        """

    # Perfil familia del paciente
    if p['prediccion_riesgo_alto']:
        return """
        DATOS DEL PACIENTE: el modelo identifica varios factores que requieren especial
        atención y seguimiento cercano por parte del equipo médico. No dispones de ninguna
        cifra ni porcentaje, no los menciones ni los inventes. Sé cercano y honesto sin
        generar alarma: reconoce que conviene un seguimiento atento, transmite que el equipo
        médico está para acompañar a la familia en cada paso, y anímales explícitamente a
        hablar con el oncólogo sobre el pronóstico y las opciones de tratamiento, que es quien
        debe dar esa información con el contexto clínico completo que tú no tienes.
        """
    else:
        return """
        DATOS DEL PACIENTE: con la información disponible, el modelo no identifica factores
        de alto riesgo. No dispones de ninguna cifra ni porcentaje, no los menciones ni los
        inventes. Puedes transmitir esto de forma esperanzadora y cercana, recordando siempre
        que el seguimiento del equipo médico sigue siendo la referencia principal.
        """



# ---------------------------------------------- Grafo RAG (Secciones 5 y 7) -------------------------------------------------

class RAGState(TypedDict):
    question: str
    perfil: str  # "oncologo" o "familia"
    prediccion: Optional[dict]
    docs: list
    answer: str

def retrieve_node(state: RAGState) -> RAGState:
    docs = qdrant.similarity_search(state['question'], k=8)
    state["docs"] = docs
    return state

def postfiltering_node(state: RAGState) -> RAGState:
    final_docs = []
    for doc in state['docs']:
        inputs = {"messages": [
            ("system", f"""
            sobre este documento:
            ----------
            {doc.page_content}
            --------------
            """),
            ("human", 
            f"""{state['question']}, dime con la etiqueta [RELEVANT] si el documento es relevante para la pregunta o [NOT RELEVANT] si el contenido no tiene nada que ver.""")
        ]}
        messages_langchain = convert_to_openai_messages(inputs['messages'])
        resp = llm.invoke(messages_langchain).content
        if "[RELEVANT]" in resp:
            final_docs.append(doc)
    state['docs'] = final_docs
    return state

def generate_node(state: RAGState) -> RAGState:
    contexto_prediccion = formatear_contexto_prediccion(state.get("prediccion"), state["perfil"])

    if state["perfil"] == "oncologo":
        tono = """
        Responde como un asistente clínico dirigido a un oncólogo pediátrico. Puedes usar
        terminología médica sin simplificarla. Si hay predicción del modelo disponible,
        indica la probabilidad exacta y el umbral de decisión usado, y recuerda brevemente
        que es una estimación de apoyo, no un diagnóstico, basada en datos de registro
        poblacional (SEER) sin variables clínicas finas (biopsia, imagen, comorbilidades).
        """
    else:
        tono = """
        Responde como asistente dirigido a la familia de un paciente pediátrico que no cuentan con
        conocimientos médicos previos. Usa lenguaje sencillo y empático, evita jerga médica
        sin explicarla. Nunca menciones cifras o porcentajes de riesgo, aunque los conocieras, ya que 
        no los tienes disponibles para este perfil.
        """

    inputs = {"messages": [
        ("system", f"""
        Eres un asistente que responde preguntas sobre osteosarcoma y sarcoma de Ewing
        pediátrico. Tienes estas fuentes de información:

        {GLOSARIO_VARIABLES}

        FUENTE 1 — Documentos de referencia general:
        ----------
        {state['docs']}
        --------------

        FUENTE 2 — Datos específicos del paciente actual (si están presentes, úsalos):
        {contexto_prediccion if contexto_prediccion else "No se han proporcionado datos de un paciente concreto en esta consulta."}

        {tono}
        """),
        ("human", f"""{state['question']}
        1. Responde en español, de forma clara.
        2. Si la pregunta trata sobre pronóstico/riesgo/supervivencia y tienes datos del
           paciente (FUENTE 2), inclúyelos siempre en tu respuesta.
        3. Si no ves documentos relevantes en la FUENTE 1 para el resto de la pregunta,
           dilo explícitamente en vez de inventar información.
        4. Nunca presentes la predicción del modelo como un diagnóstico definitivo.""")
    ]}
    messages_langchain = convert_to_openai_messages(inputs['messages'])
    resp = llm.invoke(messages_langchain).content
    state['answer'] = resp
    return state

def _build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("postfiltering", postfiltering_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "postfiltering")
    graph.add_edge("postfiltering", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag = _build_graph()