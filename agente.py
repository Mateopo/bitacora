
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory

modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,   # más bajo para respuestas más precisas y consistentes
    max_tokens=512
)

template = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de registro de incidentes técnicos.

Tu única función es interpretar lo que escribe el usuario y extraer:
- El número de caso
- El cliente según el prefijo del caso
- La descripción del incidente
- La solución aplicada (si la menciona)

REGLAS DE DETECCIÓN DE CLIENTE POR PREFIJO:
- Si el caso empieza con "IR" → Cliente: Colmedica

CÓMO DETECTAR LA SOLUCIÓN:
El usuario puede escribirla de distintas formas:
- "IR2313123 - apolo falla - solución: depuré las sesiones"
- "IR2313123 - apolo falla | fix: reinicié el servicio"
- "IR2313123 - apolo falla - se resolvió borrando caché"

Palabras clave que indican solución: "solución:", "fix:", "se resolvió",
"se solucionó", "se arregló", "corrección:", "resolved:", "se depuró"

Si no hay solución mencionada, deja ese campo como "Pendiente".

FLUJO:
1. El usuario escribe el incidente
2. Tú respondes SIEMPRE con este formato exacto:

✅ Caso registrado:
- Cliente: Colmedica
- Caso: IR2313123
- Descripción: apolo falla
- Solución: Pendiente

¿Los datos son correctos? Responde SÍ para confirmar o corrígeme.

3. Si el usuario confirma con "sí", "si", "correcto", "ok" o similar, respondes
   EXACTAMENTE con esta línea (sin cambiar nada):
   GUARDADO:Cliente=Colmedica|Caso=IR2313123|Descripción=apolo falla|Solución=Pendiente

4. Si el usuario corrige algo, actualizas los datos y vuelves a mostrar el resumen.

Responde siempre en español, breve y claro."""),
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


import httpx  # agregar este import arriba

WEBHOOK_N8N = "https://mpacheco.app.n8n.cloud/webhook/buscar-película"

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

        # ── Llamar a n8n directamente desde la API ──────────
        try:
            payload = {
                "fecha":       __import__('datetime').datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p"),
                "session_id":  session_id,
                "cliente":     datos_guardado.get("Cliente",     "No detectado"),
                "caso":        datos_guardado.get("Caso",        "No detectado"),
                "descripcion": datos_guardado.get("Descripción", "No detectada"),
                "solucion":    datos_guardado.get("Solución",    "Pendiente")
            }
            httpx.post(WEBHOOK_N8N, json=payload, timeout=10)
        except Exception:
            pass  # Si n8n falla, no interrumpe la respuesta al usuario

    return {
        "respuesta":           respuesta,
        "session_id":          session_id,
        "mensajes_en_memoria": len(historial_redis.messages),
        "guardado":            guardado,
        "datos_guardado":      datos_guardado
    }
