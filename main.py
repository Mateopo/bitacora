
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agente import chatear, obtener_historial

app = FastAPI(
    title="Agente de Registro de Incidentes",
    description="API con memoria persistente en Redis para gestión de incidentes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class Consulta(BaseModel):
    session_id: str | None = None
    mensaje:    str


class Respuesta(BaseModel):
    session_id:          str
    respuesta:           str
    mensajes_en_memoria: int
    guardado:            bool        # True cuando el usuario confirmó
    datos_guardado:      dict        # los campos extraídos listos para el webhook


@app.get("/")
def raiz():
    return {"estado": "activo", "agente": "Registro de Incidentes"}


@app.post("/consultar", response_model=Respuesta)
def consultar(consulta: Consulta):
    session_id = consulta.session_id or str(uuid.uuid4())

    resultado = chatear(
        session_id=session_id,
        mensaje=consulta.mensaje
    )

    return Respuesta(
        session_id=          session_id,
        respuesta=           resultado["respuesta"],
        mensajes_en_memoria= resultado["mensajes_en_memoria"],
        guardado=            resultado["guardado"],
        datos_guardado=      resultado["datos_guardado"]
    )


@app.delete("/limpiar/{session_id}")
def limpiar(session_id: str):
    obtener_historial(session_id).clear()
    return {"mensaje": f"Historial '{session_id}' eliminado"}
