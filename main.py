from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

# ----------------- Inicialización -----------------
app = FastAPI(title="Fast Local Summarizer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("static/index.html")


print("Cargando modelo de resumen... Esto puede tardar unos segundos.")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ----------------- Modelos -----------------
class SummarizeRequest(BaseModel):
    urls: list[str] = []
    texts: list[str] = []
    max_words: int = 80

# ----------------- Funciones -----------------
def obtener_html(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al obtener la página {url}: {e}")

def limpiar_texto(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    return " ".join(paragraphs).strip()

def resumir_texto(texto):
    """Genera resumen completo usando bloques"""
    try:
        chunk_size = 1500
        summaries = []
        for i in range(0, len(texto), chunk_size):
            chunk = texto[i:i+chunk_size]
            result = summarizer(chunk, max_length=500, min_length=60, do_sample=False)
            summaries.append(result[0]['summary_text'])
        return " ".join(summaries)
    except Exception as e:
        return f"Error resumiendo: {e}"

def recortar_resumen_coherente(resumen_completo, max_words):
    """Recorta el resumen respetando oraciones completas y el límite de palabras"""
    sentences = re.split(r'(?<=[.!?]) +', resumen_completo)
    palabras = 0
    resumen_final = []

    for s in sentences:
        cuenta_palabras = len(s.split())
        if palabras + cuenta_palabras <= max_words:
            resumen_final.append(s)
            palabras += cuenta_palabras
        else:
            break

    return " ".join(resumen_final) if resumen_final else " ".join(resumen_completo.split()[:max_words])

# ----------------- Endpoint -----------------
@app.post("/summarize/")
def summarize(request: SummarizeRequest):
    resultados = []

    # URLs
    for url in request.urls:
        try:
            html = obtener_html(url)
            texto = limpiar_texto(html)
            resumen_completo = resumir_texto(texto)
            resumen_final = recortar_resumen_coherente(resumen_completo, request.max_words)
            resultados.append({
                "input": url,
                "resumen_completo": resumen_completo,
                "resumen_final": resumen_final
            })
        except Exception as e:
            resultados.append({
                "input": url,
                "resumen_completo": f"Error: {e}",
                "resumen_final": f"Error: {e}"
            })

    # Textos directos
    for i, texto in enumerate(request.texts, start=1):
        resumen_completo = resumir_texto(texto)
        resumen_final = recortar_resumen_coherente(resumen_completo, request.max_words)
        resultados.append({
            "input": f"Text {i}",
            "resumen_completo": resumen_completo,
            "resumen_final": resumen_final
        })

    return {"results": resultados}

# ----------------- Main -----------------
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
