
# ─────────────────────────────────────────────────────────────────────
# agente.py
# Contiene toda la lógica del agente: el modelo, la memoria en Redis
# y la función principal que procesa cada mensaje del usuario.
#
# Este archivo es independiente de la API — si mañana cambias FastAPI
# por otro framework, este archivo no necesita modificaciones.
# ─────────────────────────────────────────────────────────────────────

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory


# ── El modelo ─────────────────────────────────────────────────────────
# Lo instanciamos una sola vez al arrancar el servidor.
# Si lo creáramos dentro de la función chatear(), se reconectaría
# con Groq en cada mensaje — más lento y más costoso.
modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,   # 0.7 = respuestas naturales y variadas
    max_tokens=512     # respuestas cortas — es un chat de películas
)


# ── El template ───────────────────────────────────────────────────────
# Define cómo se arma el prompt antes de enviarlo al modelo.
# Tiene tres partes:
#   1. system            → instrucciones de comportamiento del agente
#   2. MessagesPlaceholder → aquí se insertan los mensajes anteriores de Redis
#   3. human             → el mensaje actual del usuario
#
# Sin MessagesPlaceholder el agente no tendría memoria —
# cada mensaje llegaría al modelo sin contexto previo.
template = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente experto en recomendaciones de películas.
    Conoces todo tipo de géneros, directores y actores.
    Recuerdas las preferencias que el usuario ha mencionado durante
    la conversación y las usas para hacer recomendaciones personalizadas.
    Respondes de forma breve y amigable, máximo 3 líneas."""),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{mensaje}")
])


# ── Ventana de memoria ────────────────────────────────────────────────
# Cuántos mensajes del historial enviamos al modelo en cada llamada.
# 10 mensajes = 5 turnos completos (5 preguntas + 5 respuestas).
# Sin este límite, el historial crecería indefinidamente
# y el costo de tokens subiría con cada mensaje.
VENTANA = 10


def obtener_historial(session_id: str) -> RedisChatMessageHistory:
    """
    Conecta con Redis y retorna el historial de un usuario específico.

    Cada usuario tiene su propio session_id — una clave única que
    identifica su conversación en Redis. Si el usuario es nuevo,
    Redis crea un historial vacío automáticamente.

    ttl=3600 → la conversación se borra si no hay actividad en 1 hora.
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.environ["REDIS_URL"],  # Render lee esta variable automáticamente
        ttl=3600
    )


def chatear(session_id: str, mensaje: str) -> dict:
    """
    Función principal del agente.
    Recibe el mensaje del usuario y retorna la respuesta.

    Flujo interno:
        1. Lee el historial de este usuario desde Redis
        2. Toma solo los últimos VENTANA mensajes
        3. Arma el prompt y genera la respuesta con el modelo
        4. Guarda el turno nuevo en Redis
        5. Retorna la respuesta con métricas

    Parámetros:
        session_id → identifica al usuario en Redis
        mensaje    → lo que escribió el usuario

    Retorna:
        dict con la respuesta, el session_id y cuántos mensajes
        hay guardados en Redis para este usuario
    """

    # Paso 1 — traemos el historial de este usuario desde Redis
    historial_redis = obtener_historial(session_id)

    # Paso 2 — aplicamos la ventana: solo los últimos N mensajes
    # [-VENTANA:] es Python estándar para "dame los últimos N elementos"
    mensajes_recientes = historial_redis.messages[-VENTANA:]

    # Paso 3 — construimos la chain y generamos la respuesta
    # template  → arma el prompt con el historial y el mensaje actual
    # modelo    → genera la respuesta
    # StrOutputParser → convierte la respuesta a texto plano
    chain     = template | modelo | StrOutputParser()
    respuesta = chain.invoke({
        "historial": mensajes_recientes,
        "mensaje":   mensaje
    })

    # Paso 4 — guardamos este turno en Redis
    # En el próximo mensaje, estos dos líneas estarán en el historial
    historial_redis.add_user_message(mensaje)
    historial_redis.add_ai_message(respuesta)

    # Paso 5 — retornamos el resultado
    return {
        "respuesta":           respuesta,
        "session_id":          session_id,
        "mensajes_en_memoria": len(historial_redis.messages)
    }
