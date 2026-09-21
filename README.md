# Sistema de Apoyo a la Decisión Clínica en Oncología Pediátrica mediante Machine Learning y RAG

Trabajo de Fin de Máster — Máster en Big Data, Inteligencia Artificial y Data Science (UCM)
**Autor:** Alonso Castañón González

Sistema que integra un modelo predictivo de supervivencia a 5 años (explicable con SHAP) y un sistema conversacional RAG, aplicado a dos tumores óseos pediátricos: **osteosarcoma** y **sarcoma de Ewing**, a partir de datos del registro **SEER** (National Cancer Institute).

> *"Este sistema no sustituye el criterio de un oncólogo, ni tampoco la conversación de una familia con su equipo médico. Su valor está en apoyar a los dos. El trato humano sigue siendo, y creo que debe seguir siendo siempre, insustituible."*

---

## Índice

- [Motivación](#motivación)
- [Arquitectura](#arquitectura)
- [Resultados del modelo](#resultados-del-modelo)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Stack tecnológico](#stack-tecnológico)
- [Datos](#datos)
- [Instalación y uso](#instalación-y-uso)
- [API REST](#api-rest)
- [Diseño de seguridad clínica](#diseño-de-seguridad-clínica)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Memoria completa](#memoria-completa)
- [Licencia](#licencia)

---

## Motivación

El osteosarcoma y el sarcoma de Ewing son los tumores óseos más frecuentes en niños y adolescentes. Hoy en día su pronóstico se apoya principalmente en la experiencia del equipo médico, sin herramientas cuantitativas ampliamente adoptadas que integren de forma sistemática variables como el estadio, la localización o la respuesta al tratamiento. Al mismo tiempo, las familias afrontan un proceso de diagnóstico largo y complejo, con terminología desconocida y una carga emocional muy alta, por lo que el acceso a información clara y fiable es clave.

Este trabajo diseña un sistema que:
1. Construye un **pipeline reproducible** de limpieza y preparación de datos sobre SEER.
2. Realiza un **análisis exploratorio exhaustivo** que orienta el modelado.
3. Entrena y compara varios modelos de ML, **calibrando el umbral de decisión** según el coste clínico de cada tipo de error.
4. Dota al modelo final de **explicabilidad** mediante SHAP.
5. Diseña un sistema **RAG** que adapta contenido y tono según el perfil de quien pregunta (oncólogo o familia).
6. Integra ambas capas en una **API REST** productivizada, con una interfaz de demostración.

## Arquitectura

```
                         ┌───────────────────────────┐
                         │        SEER Research       │
                         │   (4.915 pacientes ped.)   │
                         └─────────────┬──────────────┘
                                        │
                     01_limpieza.ipynb │ 02_EDA.ipynb
                                        ▼
                         ┌───────────────────────────┐
                         │     Dataset limpio         │
                         │   (CSV + Parquet)          │
                         └─────────────┬──────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                        ▼
     ┌──────────────────────────┐            ┌──────────────────────────────┐
     │   CAPA PREDICTIVA (ML)   │            │        CAPA RAG               │
     │ 03_modelo_ML.ipynb       │            │ 04_RAG_pipeline.ipynb          │
     │                          │            │                                │
     │ Dummy / LR / RF / XGBoost│            │ 8 documentos NCI → chunks      │
     │ Umbral calibrado (F-beta)│            │ → embeddings multilingües      │
     │ Explicabilidad (SHAP)    │            │ → Qdrant (183 fragmentos)      │
     │                          │            │                                │
     │ models/xgboost_final.pkl │            │ LangGraph: retrieve →          │
     │ esquema_categorias.json  │            │   postfiltering → generate     │
     └─────────────┬────────────┘            └───────────────┬────────────────┘
                    │                                         │
                    └───────────────────┬─────────────────────┘
                                         ▼
                         ┌───────────────────────────┐
                         │     src/rag.py             │
                         │  (predicción + RAG unidos) │
                         └─────────────┬──────────────┘
                                        ▼
                         ┌───────────────────────────┐
                         │   src/main.py (FastAPI)    │
                         │  POST /predict  POST /ask  │
                         │  GET  /health               │
                         └─────────────┬──────────────┘
                                        ▼
                         ┌───────────────────────────┐
                         │   frontend/index.html       │
                         │   Demo chat (oncólogo/       │
                         │   familia)                   │
                         └───────────────────────────┘
```

## Resultados del modelo

Comparación de modelos (umbral 0.5):

| Modelo    | PR-AUC | ROC-AUC |
|-----------|--------|---------|
| Dummy     | 0.295  | 0.516   |
| LogReg    | 0.518  | 0.736   |
| RF        | 0.559  | 0.737   |
| **XGBoost** | **0.561** | **0.745** |

El umbral de decisión final se recalibró mediante **F-beta (β=1.5)**, priorizando el recall por el mayor coste clínico de un falso negativo (no detectar a un paciente de riesgo) frente a un falso positivo:

| Umbral | Recall | PR-AUC | ROC-AUC |
|--------|--------|--------|---------|
| 0.50 (por defecto) | 0.280 | — | — |
| **0.18 (calibrado)** | **0.875** | 0.561 | 0.745 |

**Explicabilidad (SHAP):** `stage` es, con diferencia, la variable con mayor peso en la predicción, seguida de `tumor_type` y `year_diagnosis` (los años más recientes reducen el riesgo, coherente con la mejora de los tratamientos con el tiempo).

**Supervivencia observada (Kaplan-Meier):** 68.4% a 5 años en la cohorte global.

## Estructura del repositorio

```
├── notebooks/
│   ├── 01_limpieza.ipynb          # Carga, limpieza, tipificación, variable objetivo
│   ├── 02_EDA.ipynb               # Análisis exploratorio
│   ├── 03_modelo_ML.ipynb         # Modelado, calibración de umbral, SHAP
│   └── 04_RAG_pipeline.ipynb      # Scraping NCI, chunking, Qdrant, LangGraph RAG
├── src/
│   ├── rag.py                     # Lógica de predicción + grafo RAG (LangGraph)
│   └── main.py                    # API FastAPI (/predict, /ask, /health)
├── frontend/
│   └── index.html                 # Demo de chat conectada a la API
├── models/
│   ├── xgboost_final.pkl          # Modelo final entrenado
│   ├── esquema_categorias.json    # Esquema de categorías/rangos (orden de entrenamiento)
│   └── config_rag.json            # Configuración del sistema RAG
├── data/
│   └── splits/                    # train.csv / test.csv
├── memoria/
│   └── Alonso_Castañon_Gonzalez_Memoria.pdf
├── requirements.txt
└── README.md
```

## Stack tecnológico

- **Datos y ML:** pandas, scikit-learn (Pipeline, ColumnTransformer, GridSearchCV), XGBoost (`<3.0.0`, fijado por compatibilidad con SHAP), SHAP, lifelines (Kaplan-Meier, log-rank)
- **RAG:** LangChain, LangGraph (grafo de 3 nodos: retrieve → postfiltering → generate), Qdrant (vector DB, Docker), embeddings multilingües HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`), OpenAI `gpt-4o-mini`
- **API y frontend:** FastAPI, Pydantic, Uvicorn, HTML/CSS/JS vanilla
- **Otros:** BeautifulSoup (scraping de documentación del NCI), python-dotenv

## Datos

Los datos proceden del **registro SEER** (Surveillance, Epidemiology, and End Results) del National Cancer Institute, bajo un acuerdo de uso de datos (DUA) que restringe su redistribución. Este repositorio **no incluye** el extracto original de SEER*Stat; los notebooks documentan el proceso completo de obtención, limpieza y transformación para permitir su reproducción por quien tenga acceso autorizado al registro.

Puntos clave del proceso de limpieza (`01_limpieza.ipynb`):
- SEER codifica los desconocidos como el string `Blank(s)`, no como nulo estándar.
- Los valores ausentes en `stage`, `tumor_size_cs` y `surgery_code` no son aleatorios (MCAR), sino que siguen un mecanismo **MAR** ligado al año de diagnóstico y a cambios en los protocolos de codificación de SEER a lo largo del tiempo — se trataron de forma diferenciada por variable en lugar de eliminar filas.
- Corrección de un cambio de codificación (2007) que separaba casos de sarcoma de Ewing bajo distintos códigos histológicos (9260, 9364, 9365), unificados por tratarse de la misma entidad biológica.
- Muestra final: **4.915 pacientes**, variable objetivo `target` (no supervivencia a 5 años): 1.395 positivos (28,4%) / 3.520 negativos (71,6%).

## Instalación y uso

```bash
# 1. Clonar y crear entorno
git clone <url-del-repositorio>
cd <repositorio>
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con la API key de OpenAI, host/puerto de Qdrant, etc.

# 4. Levantar Qdrant (vector DB) con Docker
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# 5. Ejecutar los notebooks en orden (generan modelos/índices necesarios para la API)
#    01_limpieza.ipynb → 02_EDA.ipynb → 03_modelo_ML.ipynb → 04_RAG_pipeline.ipynb

# 6. Levantar la API
uvicorn src.main:app --reload

# 7. Abrir la demo
#    Abrir frontend/index.html en el navegador (la API debe estar corriendo en localhost)
```

> Los notebooks deben ejecutarse en orden la primera vez: generan el dataset limpio, el modelo entrenado (`models/xgboost_final.pkl`), el esquema de categorías y la colección de Qdrant que `src/rag.py` necesita para arrancar.

## API REST

| Método | Endpoint    | Descripción                                                                 |
|--------|-------------|------------------------------------------------------------------------------|
| POST   | `/predict`  | Predicción directa de supervivencia a 5 años a partir de los datos del paciente |
| POST   | `/ask`      | Consulta en lenguaje natural al sistema RAG (perfil `oncologo` o `familia`)  |
| GET    | `/health`   | Estado del servicio                                                          |

## Diseño de seguridad clínica

El sistema distingue dos perfiles de usuario con un tratamiento diferenciado de la probabilidad de supervivencia:

- **Perfil `oncologo`:** recibe la cifra exacta de probabilidad, con su explicación SHAP.
- **Perfil `familia`:** **nunca** recibe la cifra exacta. Esta restricción se implementa en **dos capas independientes**, no como una simple instrucción de prompt:
  1. La probabilidad ni siquiera se incluye en el prompt enviado al LLM para este perfil (el modelo de lenguaje no tiene acceso al dato).
  2. A nivel de API, el campo `prediccion` de la respuesta solo se rellena cuando `perfil="oncologo"`.

Este diseño de doble capa se valoró más robusto que confiar en que un LLM respete de forma consistente una instrucción de prompt. Se verificó mediante una evaluación cualitativa de 12 casos de prueba, confirmando **cero filtraciones numéricas** al perfil familia.

## Limitaciones y trabajo futuro

- Extensión a otras enfermedades oncológicas pediátricas, aprovechable gracias al etiquetado por enfermedad del sistema RAG y a la parametrización del esquema de categorías (aunque exigiría repetir el proceso de limpieza, EDA y entrenamiento para cada nueva patología).
- Evaluación cuantitativa formal del sistema RAG con métricas estandarizadas.
- Incorporación de memoria conversacional entre turnos.
- Exploración de un diseño de agente con herramientas (descartado en este trabajo por tiempo y por el riesgo de un comportamiento menos predecible en un contexto clínico sensible).
- Cualquier uso clínico real exigiría validación externa y prospectiva del modelo, junto con los procesos de revisión regulatoria y supervisión clínica correspondientes — fuera del alcance de este trabajo.
- El frontend de demostración usa opciones de formulario escritas a mano a partir del esquema de categorías vigente en el momento de su creación; si el modelo se reentrena y el esquema cambia, las opciones deben actualizarse manualmente.

## Memoria completa

Este README es un resumen orientado a GitHub. La metodología completa (estado del arte, limpieza y preparación de datos, EDA, capa ML, capa RAG, productivización, resultados, discusión y conclusiones) está documentada en la memoria del TFM: `memoria/Alonso_Castañon_Gonzalez_Memoria.pdf`.

## Licencia

Trabajo académico presentado como Trabajo de Fin de Máster (UCM). Uso de los datos SEER sujeto al acuerdo de uso de datos (DUA) del National Cancer Institute; consultar [seer.cancer.gov](https://seer.cancer.gov) para las condiciones de acceso y redistribución.
