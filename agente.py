%%writefile agente_peliculas/agente.py

import os
import httpx
import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory

modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=512
)

WEBHOOK_N8N = "hhttps://mpacheco.app.n8n.cloud/webhook/buscar-película"

template = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de registro de incidentes técnicos.

Cuando el usuario escriba un incidente, extrae los datos y guárdalos 
INMEDIATAMENTE sin pedir confirmación.

REGLAS DE DETECCIÓN DE CLIENTE POR PREFIJO:
- Si el caso empieza con "IR" → Cliente: Colmedica

CÓMO DETECTAR LA SOLUCIÓN:
Palabras clave: "solución:", "fix:", "se resolvió", "se depuró",
"se reinició", "se arregló", "corrección:", "resolved:"
Si no hay solución mencionada, usa "Pendiente".

INSTRUCCIÓN CRÍTICA:
Cuando recibas un incidente responde ÚNICAMENTE con esta línea exacta,
sin texto adicional, sin explicaciones, sin emojis:
GUARDADO:Cliente=VALOR|Caso=VALOR|Descripción=VALOR|Solución=VALOR

Ejemplo:
Usuario escribe: IR12345 - apolo falla - se depuran sesiones
Tu respuesta: GUARDADO:Cliente=Colmedica|Caso=IR12345|Descripción=apolo falla|Solución=se depuran sesiones

Usuario escribe: IR99887 - error en login
Tu respuesta: GUARDADO:Cliente=Colmedica|Caso=IR99887|Descripción=error en login|Solución=Pendiente"""),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{mensaje}")
])

VENTANA = 10


def obtener_historial(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.environ["REDIS_URL"],
        ttl=3600
    )


def chatear(session_id: str, mensaje: str) -> dict:
    historial_redis = obtener_historial(session_id)
    mensajes_recientes = historial_redis.messages[-VENTANA:]

    chain     = template | modelo | StrOutputParser()
    respuesta = chain.invoke({
        "historial": mensajes_recientes,
        "mensaje":   mensaje
    })

    historial_redis.add_user_message(mensaje)
    historial_redis.add_ai_message(respuesta)

    guardado = False
    datos_guardado = {}

    if respuesta.startswith("GUARDADO:"):
        guardado = True
        partes = respuesta.replace("GUARDADO:", "").split("|")
        for parte in partes:
            if "=" in parte:
                clave, valor = parte.split("=", 1)
                datos_guardado[clave.strip()] = valor.strip()

        # Llamar a n8n directamente desde la API
        try:
            payload = {
                "fecha":       datetime.datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p"),
                "session_id":  session_id,
                "cliente":     datos_guardado.get("Cliente",     "No detectado"),
                "caso":        datos_guardado.get("Caso",        "No detectado"),
                "descripcion": datos_guardado.get("Descripción", "No detectada"),
                "solucion":    datos_guardado.get("Solución",    "Pendiente")
            }
            httpx.post(WEBHOOK_N8N, json=payload, timeout=10)
        except Exception:
            pass

    # Respuesta amigable al usuario cuando se guarda
    respuesta_usuario = respuesta
    if guardado:
        respuesta_usuario = (
            f"✅ Incidente registrado correctamente:\n"
            f"- Cliente: {datos_guardado.get('Cliente', '—')}\n"
            f"- Caso: {datos_guardado.get('Caso', '—')}\n"
            f"- Descripción: {datos_guardado.get('Descripción', '—')}\n"
            f"- Solución: {datos_guardado.get('Solución', '—')}"
        )

    return {
        "respuesta":           respuesta_usuario,
        "session_id":          session_id,
        "mensajes_en_memoria": len(historial_redis.messages),
        "guardado":            guardado,
        "datos_guardado":      datos_guardado
    }
