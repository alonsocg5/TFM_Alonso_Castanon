"""
src/main.py

API REST del sistema (POST /predict, POST /ask).

Arrancar con: uvicorn src.main:app --reload
Demo funcional del prototipo: http://localhost:8000/docs

"""

from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from . import rag

app = FastAPI(
    title="Sistema de Apoyo a la Decisión Clínica en Oncología Pediátrica",
    description="API del TFM: predicción de supervivencia (XGBoost) y asistente conversacional (RAG).",
    version="1.0.0",
)


# ----------------------------------- Esquemas Pydantic ---------------------------------------

# Definen la forma y validación de las peticiones/respuestas de la API. FastAPI usa estas clases para: 
#   1) Validar automáticamente lo que llega en cada petición, devolviendo un error 422 si algo no es del tipo esperado. 
#   2) Generar la documentación interactiva de /docs sin tener que escribirla a mano. 
#   3) Convertir automáticamente entre JSON y objetos Python en ambas direcciones.
#
# Las descripciones de cada campo categórico se generan dinámicamente a partir de esquema_categorias.json (cargado por rag.py), para que /docs
# siempre muestre las opciones válidas reales del modelo, sin tener que mantenerlas duplicadas a mano en dos archivos distintos.

# -----------------------------------------------------------------------------------------------

class DatosPaciente(BaseModel):
    age_group: str = Field(..., description=f"Opciones: {rag.esquema['categoricas']['age_group']}")
    sex: str = Field(..., description=f"Opciones: {rag.esquema['categoricas']['sex']}")
    tumor_type: str = Field(..., description=f"Opciones: {rag.esquema['categoricas']['tumor_type']}")
    primary_site: str = Field(..., description=f"Opciones: {rag.esquema['categoricas']['primary_site']}")
    stage: Optional[str] = Field(None, description=f"Opciones: {rag.esquema['categoricas']['stage']}. Omitir si no está registrado.")
    surgery_code: str = Field(..., description=f"Opciones: {rag.esquema['categoricas']['surgery_code']}")
    radiation: str = Field(..., description=f"Opciones: {rag.esquema['categoricas']['radiation']}")
    chemotherapy: str = Field(..., description=f"Opciones: {rag.esquema['categoricas']['chemotherapy']}")
    year_diagnosis: int = Field(..., description="Año de diagnóstico")


class PrediccionResponse(BaseModel):
    probabilidad_no_supervivencia_5a: float
    prediccion_riesgo_alto: bool
    umbral_usado: float


class AskRequest(BaseModel):
    question: str
    perfil: Literal["oncologo", "familia"]
    paciente: Optional[DatosPaciente] = None

# Solo se incluye la predicción del paciente si el perfil es el del oncólogo. Para el de familia, este campo queda vacío aunque haya paciente,
# así el JSON de respuesta nunca filtra la cifra, ni siquiera fuera del texto.
class AskResponse(BaseModel):
    answer: str
    prediccion: Optional[PrediccionResponse] = None


# ------------------------------- Endpoints ----------------------------------------

@app.post("/predict", response_model=PrediccionResponse)
def predict(paciente: DatosPaciente):
    """
    Predicción del modelo XGBoost para un paciente, sin pasar por el RAG.
    Devuelve siempre la cifra exacta, pensado para uso clínico, no para el chat orientado a familias.

    Reutiliza rag.predecir_paciente(), la misma función ya probada en 04_RAG_pipeline.ipynb (Sección 6). 
    Construye el DataFrame con los dtypes exactos del entrenamiento y valida cada categoría contra esquema_categorias.json. 
    Si algún valor no es válido, se lanza un HTTP 422 con el mensaje de error, en vez de reimplementar la validación por segunda vez.
    """
    try:
        resultado = rag.predecir_paciente(paciente.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PrediccionResponse(
        probabilidad_no_supervivencia_5a=resultado["probabilidad_no_supervivencia_5a"],
        prediccion_riesgo_alto=resultado["prediccion_riesgo_alto"],
        umbral_usado=resultado["umbral_usado"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Consulta al sistema RAG (grafo LangGraph de rag.py: retrieve -> postfiltering -> generate), combinando los documentos indexados y, 
    opcionalmente, la predicción de un paciente concreto si se envían sus datos.

    El campo `prediccion` de la respuesta solo se rellena si `perfil="oncologo"`, mientras que para `perfil="familia"` se omite no solo 
    en el texto generado sino también a nivel de JSON, por seguridad clínica.

    """
    prediccion_dict = None
    if req.paciente is not None:
        try:
            prediccion_dict = rag.predecir_paciente(req.paciente.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    resultado = rag.rag.invoke({
        "question": req.question,
        "perfil": req.perfil,
        "prediccion": prediccion_dict,
    })

    prediccion_response = None
    if prediccion_dict is not None and req.perfil == "oncologo":
        prediccion_response = PrediccionResponse(
            probabilidad_no_supervivencia_5a=prediccion_dict["probabilidad_no_supervivencia_5a"],
            prediccion_riesgo_alto=prediccion_dict["prediccion_riesgo_alto"],
            umbral_usado=prediccion_dict["umbral_usado"],
        )

    return AskResponse(answer=resultado["answer"], prediccion=prediccion_response)


@app.get("/health")
def health():
    """
    Comprobación rápida de que la API y sus dependencias (Qdrant, modelo) están operativas, 
    sin necesidad de probar los endpoints completos (que consumen tokens de la API de OpenAI).
    """
    return {"status": "ok", "coleccion_qdrant": rag.COLLECTION_NAME}