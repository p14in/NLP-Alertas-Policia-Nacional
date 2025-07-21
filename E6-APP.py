import os
import re
import sqlite3
from datetime import datetime
from io import BytesIO
import pandas as pd
import streamlit as st
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

# ---------------------------------------------------------------------------
# 1.  CARGAR spaCy Y CONFIGURAR ENTITY RULER
# ---------------------------------------------------------------------------

nlp = spacy.load("es_core_news_sm")

# Añadimos el EntityRuler ANTES de "ner" y sobrescribimos entidades
ruler = nlp.add_pipe(
    "entity_ruler",
    before="ner",
    config={"overwrite_ents": True}
)

# ----------------------------- diccionarios -------------------------------

diccionario_bandas = [
    "bandoleros", "caballeros oscuros", "capos", "chonekiller", "choneros",
    "corvicheros", "cuartel de las feas", "cubanos", "fatales", "gángster",
    "kater piler", "lagartos", "latin kings", "lobos", "mafia 18",
    "maras salvatruchas", "pandilleros", "pistoleros", "tiguerones",
    "águilas", "águilaskiller"
]

diccionario_grupos = [
    "aad", "al murabitoun", "al qaeda", "al-nusra", "al-shabaab", "anf/hts",
    "ansar al-shari'a", "ansaru", "aoi", "bacrim", "boko haram",
    "brigadas al-ashtar", "cartel de sinaloa", "cpp/npa", "dhkp/c", "eln",
    "eta", "farc", "farc-ep", "hamas", "harakat sawa'd misr",
    "hizbul mujahideen", "hizbula", "hqn", "huji-b", "isis-bangladesh",
    "isis-filipinas", "isis-gs", "isis-k", "isis-libia", "isis-mozambique",
    "isis-áfrica occidental", "jat", "jaysh al adl", "jemaah anshorut tauhid",
    "jrtn", "lashkar-e-taiba", "ltte", "pflp-gc", "pkk",
    "provincia del sinaí", "red haqqani", "segunda marquetalia",
    "sendero luminoso", "talibanes", "urabeños"
]

diccionario_armas = [
    "ak-47", "ametralladora", "armas", "escopeta", "fusil", "glock",
    "m16", "m4", "mp5", "pertrechos", "pistola", "revolver", "rifle",
    "smith&wesson", "taurus", "uzi"
]

diccionario_explosivos = [
    "c4", "detonador", "dinamita", "explosivo", "granada", "hexógeno",
    "nitroglicerina", "pentolita", "rx8", "semtex", "termita", "tnt"
]

# helper: convierte frase en lista de tokens LOWER
def frase_a_patron(frase: str):
    return [{"LOWER": tok} for tok in frase.split()]

# patrones iniciales
patterns = [
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]},
    {"label": "TIME", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}:\d{2}"}}]},
]

# añadir diccionarios
for palabra in diccionario_bandas:
    patterns.append({"label": "BANDA", "pattern": frase_a_patron(palabra)})

for palabra in diccionario_grupos:
    patterns.append({"label": "GRUPO", "pattern": frase_a_patron(palabra)})

for palabra in diccionario_armas:
    patterns.append({"label": "ARMA", "pattern": frase_a_patron(palabra)})

for palabra in diccionario_explosivos:
    patterns.append({"label": "EXPLOSIVO", "pattern": frase_a_patron(palabra)})

# ubicaciones compuestas frecuentes
ubicaciones_compuestas = [
    "vía a daule", "la fortuna", "puente viejo", "cima del coloso"
]
for loc in ubicaciones_compuestas:
    patterns.append({"label": "LOC", "pattern": frase_a_patron(loc)})

ruler.add_patterns(patterns)

# ---------------------------------------------------------------------------
# 2.  CARGAR MODELOS DE SENTIMIENTO Y PRIORIDAD
# ---------------------------------------------------------------------------

MODEL_DIR = "models"
try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "sentiment_model"))
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "sentiment_model"))
    priority_tokenizer  = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "best_model"))
    priority_model      = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "best_model"))
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)
priority_model.to(device)

# ---------------------------------------------------------------------------
# 3.  PRIORIDADES Y MAPEOS
# ---------------------------------------------------------------------------

priority_class_map = {
    0: "affected_individual", 1: "caution_and_advice", 2: "displaced_and_evacuations",
    3: "donation_and_volunteering", 4: "infrastructure_and_utilities_damage",
    5: "injured_or_dead_people", 6: "missing_and_found_people", 7: "not_humanitarian",
    8: "requests_or_needs", 9: "response_efforts", 10: "sympathy_and_support"
}

priority_name_map = {
    "affected_individual": "Persona afectada",
    "caution_and_advice": "Precaución",
    "displaced_and_evacuations": "Desplazados y evacuaciones",
    "donation_and_volunteering": "Donaciones y voluntariado",
    "infrastructure_and_utilities_damage": "Posible daño a infraestructura o servicios",
    "injured_or_dead_people": "Personas heridas o muertas",
    "missing_and_found_people": "Personas desaparecidas",
    "not_humanitarian": "No humanitario",
    "requests_or_needs": "Solicitudes o necesidades",
    "response_efforts": "Esfuerzos de respuesta",
    "sympathy_and_support": "Simpatía y apoyo"
}

alert_classes = {
    "missing_and_found_people", "injured_or_dead_people",
    "infrastructure_and_utilities_damage", "caution_and_advice",
    "affected_individual"
}

# ---------------------------------------------------------------------------
# 4.  BASE DE DATOS
# ---------------------------------------------------------------------------

DB_NAME = "NOVEDADES.db"

def crear_base_datos():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alertas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    oracion TEXT,
                    tipo_novedad TEXT,
                    ubicacion TEXT,
                    identidades TEXT,
                    fechas TEXT,
                    vehiculo TEXT,
                    armas_explosivos TEXT,
                    sentimiento INTEGER,
                    class_label TEXT,
                    timestamp TEXT
                )"""
            )
    except sqlite3.Error as e:
        st.error(f"Error al crear la base de datos: {e}")
        raise

def verificar_base_datos():
    if not os.path.exists(DB_NAME):
        print("Creando base de datos...")
        crear_base_datos()
# ---------------------------------------------------------------------------
# 5.  LIMPIEZA DE TEXTO
# ---------------------------------------------------------------------------

def limpiar_texto(txt: str) -> str:
    txt = re.sub(r"@[\w_]+", "", txt)
    txt = re.sub(r"#[\w-]+", "", txt)
    txt = re.sub(r"http\S+|www\.\S+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"[^\wáéíóúÁÉÍÓÚñÑüÜ¿?¡!.,:;_\-()/\s]", "", txt)
    return txt

# --------------------------------------------------------------
# 6.  HIGHLIGHT ENTITIES (para debug visual en Streamlit)
# --------------------------------------------------------------

def highlight_entities(text: str) -> str:
    doc = nlp(text)
    html = ""
    COLORS = {
        "PERSON": "#FFDD00", "NORP": "#FF9800", "FAC": "#A97560",
        "ORG": "#4CAF50", "GPE": "#BFA815", "LOC": "#0041AB",
        "PRODUCT": "#B3890E", "EVENT": "#330047", "DATE": "#BBDEFB",
        "TIME": "#476035", "BANDA": "#FF6B6B", "GRUPO": "#FF9800",
        "ARMA": "#4B8BFF", "EXPLOSIVO": "#9C27B0"
    }
    for tok in doc:
        if tok.ent_type_:
            style = f"background-color:{COLORS.get(tok.ent_type_, '#74782F')}; color:white;"
            html += f"<span style='{style} border-radius:4px; padding:2px'>{tok.text_with_ws}<sub style='font-size:8px'>{tok.ent_type_}</sub></span>"
        else:
            html += tok.text_with_ws
    return html

# --------------------------------------------------------------
# 7.  SENTIMIENTO Y PRIORIDAD
# --------------------------------------------------------------

def analyze_sentiment(sentences):
    ratings, html = [], ""
    for sent in sentences:
        if not sent:
            continue
        inputs = sentiment_tokenizer(sent, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        rating = sentiment_model(**inputs).logits.argmax(-1).item()
        ratings.append(rating)
        color0 = {0: "#FF6B6B", 1: "#F6EFA6", 2: "#6BCB77"}[rating]
        label = {0: "Negativo", 1: "Neutral", 2: "Positivo"}[rating]
        text_color = "black"  # Define text_color
        html += f"<div style='background:{color0};color:{text_color};padding:4px;border-radius:4px;margin-bottom:4px'>{sent} <strong>({label})</strong></div>"
    return ratings, html


def analyze_priority(sentences):
    results = []
    for sent in sentences:
        if not sent:
            results.append("not_humanitarian")
            continue
        inputs = priority_tokenizer(sent, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        idx = priority_model(**inputs).logits.argmax(-1).item()
        results.append(priority_class_map.get(idx, "Desconocido"))
    return results

# ---------------------------------------------------------------------------
# 8.  ANALIZAR UNA ORACIÓN Y CONSTRUIR ALERTA LIMPIA
# ---------------------------------------------------------------------------

def extraer_alerta(sent: str, sentiment: int, class_label: str):
    doc = nlp(sent)

    # diccionario de listas para BANDA/GRUPO/ARMA/EXPLOSIVO
    found = {"BANDA": [], "GRUPO": [], "ARMA": [], "EXPLOSIVO": []}
    for ent in doc.ents:
        if ent.label_ in found:
            found[ent.label_].append(ent.text.lower())

    # vehículo
    vehiculo_match = re.search(r"\b[A-Z]{3}-\d{3}\b", sent)
    vehiculo = vehiculo_match.group(0) if vehiculo_match else "N/A"

    # -------------------- UBICACIÓN (contiguas) --------------------
    loc_parts, current, last_end = [], [], -1
    for ent in [e for e in doc.ents if e.label_ in ("GPE", "LOC")]:
        if ent.start == last_end:
            current.append(ent.text)
        else:
            if current:
                loc_parts.append(" ".join(current))
            current = [ent.text]
        last_end = ent.end
    if current:
        loc_parts.append(" ".join(current))
    ubicacion = ", ".join(loc_parts) or "N/A"

    # identidades, fechas, armas/explosivos
    identidades = ", ".join({e.text for e in doc.ents if e.label_ in ["PERSON", "BANDA"]}) or "N/A"
    fechas      = ", ".join({e.text for e in doc.ents if e.label_ == "DATE"})   or "N/A"
    arm_exp     = ", ".join(sorted(set(found["ARMA"] + found["EXPLOSIVO"]))) or "N/A"

    # tipo de novedad sin duplicados
    tipo_nov = []
    for lbl in ("BANDA", "GRUPO", "ARMA", "EXPLOSIVO"):
        if found[lbl]:
            tipo_nov.append(lbl.title())
    if vehiculo_match:
        tipo_nov.append("Vehículo")
    tipo_novedad = ", ".join(tipo_nov) or "Novedad"

    return {
        "oracion": sent,
        "tipo_novedad": tipo_novedad,
        "ubicacion": ubicacion,
        "identidades": identidades,
        "fechas": fechas,
        "vehiculo": vehiculo,
        "armas_explosivos": arm_exp,
        "sentimiento": sentiment,
        "class_label": class_label
    }

def generar_tabla_prioridades(sentences, priorities):
    data = []
    for sent, priority in zip(sentences, priorities):
        genera_alerta = priority in alert_classes
        bg_color = "#E6FFE6" if genera_alerta else "#F0F0F0"
        icono = "✅" if genera_alerta else "❌"
        data.append({
            "Oración": sent,
            "Clase de Prioridad": priority_name_map.get(priority, priority),
            "¿Genera Alerta?": icono,
            "_background": bg_color
        })
    
    df = pd.DataFrame(data)
    
    def estilo_fondo(row):
        color = row.get('_background', 'black')  # Fondo negro por defecto
        return [f'background-color: {color}' for _ in row.index if row.index.name != '_background']
    
    columnas_visibles = ["Oración", "Clase de Prioridad", "¿Genera Alerta?"]
    styled_df = df[columnas_visibles].style.apply(estilo_fondo, axis=1)
    
    return styled_df, data

# ---------------------------------------------------------------------------
# 9.  PDF REPORT
# ---------------------------------------------------------------------------
def generar_reporte_pdf(alerts):
    """Genera un PDF con cabecera/pie y celdas formateadas en varias líneas."""

    # --- buffer y documento ---
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=50, bottomMargin=50,
        leftMargin=30, rightMargin=30
    )

    # --- cabecera y pie ---
    def add_header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(
            doc.width / 2 + doc.leftMargin,
            doc.pagesize[1] - 30,
            "RESERVADO"
        )
        canvas.drawCentredString(
            doc.width / 2 + doc.leftMargin,
            20,
            "RESERVADO"
        )
        canvas.restoreState()

    # --- estilos ---
    styles = getSampleStyleSheet()
    title_style   = styles["Title"]
    heading_style = styles["Heading2"]
    cell_style = ParagraphStyle(
        "cell", parent=styles["Normal"],
        fontSize=9, leading=11
    )

    # helper para garantizar dos líneas
    def ensure_two_lines(txt):
        html = txt.replace(", ", "<br/>")
        if "<br/>" not in html:
            html += "<br/>&nbsp;"
        return html

    # --- contenido ---
    elements = []
    elements.append(Paragraph(
        f"Reporte de Novedades - {datetime.now():%d/%m/%Y}",
        title_style
    ))
    elements.append(Spacer(1, 12))

    data = [
        ["ID", "Tipo de Alerta", "Ubicación", "Identidades",
         "Fechas", "Vehículo", "Armas/Explosivos"]
    ]

    for idx, alert in enumerate(alerts, start=1):
        row = [
            str(idx),
            Paragraph(ensure_two_lines(alert["tipo_novedad"]), cell_style),
            Paragraph(ensure_two_lines(alert["ubicacion"]),    cell_style),
            Paragraph(alert["identidades"].replace(", ", "<br/>") or "N/A",
                      cell_style),
            Paragraph(alert["fechas"].replace(", ", "<br/>") or "N/A",
                      cell_style),
            Paragraph(alert["vehiculo"] or "N/A", cell_style),
            Paragraph(alert["armas_explosivos"].replace(", ", "<br/>") or "N/A",
                      cell_style)
        ]
        data.append(row)

    table = Table(
        data,
        colWidths=[25, 100, 90, 90, 60, 55, 80],
        repeatRows=1
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, colors.lightgrey])
    ]))

    elements.extend([
        Paragraph("Tabla de Novedades Registradas", heading_style),
        table
    ])

    # --- render ---
    doc.build(
        elements,
        onFirstPage=add_header_footer,
        onLaterPages=add_header_footer
    )
    buffer.seek(0)
    return buffer

# Interfaz con Streamlit
def main():
    st.title("NLP Policía Nacional - Ecuador")

    # Verificar base de datos
    verificar_base_datos()

    # Inicializar estado
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'pdf_buffer' not in st.session_state:
        st.session_state.pdf_buffer = None

    # Entrada de texto con key único
    user_text = st.text_area("Ingresa texto relacionado con Ecuador", height=200, key="main_text_area")
    pdf_filename_input = st.text_input("Nombre del archivo PDF (sin extensión)", value="reporte_novedades", key="pdf_filename_input")

    if st.button("Analizar", key="analyze_button"):
        if not user_text.strip():
            st.warning("Por favor, ingresa un texto.")
            return

        text_clean = limpiar_texto(user_text)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_clean) if s.strip()]
        alerts = []

        # Análisis de sentimientos y prioridades
        sentiment_ratings, sentiment_html = analyze_sentiment(sentences)
        priorities = analyze_priority(sentences)

        # Generar tabla de prioridades
        tabla_prioridades, prioridades_data = generar_tabla_prioridades(sentences, priorities)

        # Detección de entidades y palabras clave
        for sent, sentiment, class_label in zip(sentences, sentiment_ratings, priorities):
            sent_lower = sent.lower()
            detected_keywords = []
            tipo_novedad = []

            # Buscar palabras clave usando spaCy
            doc = nlp(sent)
            for ent in doc.ents:
                if ent.label_ in ["BANDA", "GRUPO", "ARMA", "EXPLOSIVO"]:
                    detected_keywords.append(ent.text.lower())
                    tipo_novedad.append(ent.label_.capitalize())

            # Detectar vehículo
            vehiculo = re.search(r"\b[A-Z]{3}-\d{3}\b", sent)
            vehiculo_str = vehiculo.group(0) if vehiculo else "N/A"
            if vehiculo:
                detected_keywords.append(vehiculo_str)
                tipo_novedad.append("Vehículo")

            # Extracción de entidades
            ubicacion = ", ".join([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]) or "N/A"
            identidades = ", ".join([ent.text for ent in doc.ents if ent.label_ in ["PERSON"]]) or "N/A"
            fechas = ", ".join([ent.text for ent in doc.ents if ent.label_ == "DATE"]) or "N/A"
            armas_explosivos = ", ".join(sorted(set([kw for kw in detected_keywords if kw in diccionario_armas + diccionario_explosivos]))) or "N/A"

            # Depuración
            print(f"Oración: {sent}")
            print(f"Palabras clave detectadas: {detected_keywords}")
            print(f"Sentimiento: {sentiment}")
            print(f"Clase predicha: {class_label}")
            print(f"¿Cumple criterios? {detected_keywords or class_label in alert_classes}")

            # Registrar alerta si hay palabras clave o prioridad en alert_classes
            if detected_keywords or class_label in alert_classes:
                alert = {
                    "oracion": sent,
                    "tipo_novedad": ", ".join(tipo_novedad) or "Novedad",
                    "ubicacion": ubicacion,
                    "identidades": identidades,
                    "fechas": fechas,
                    "vehiculo": vehiculo_str,
                    "armas_explosivos": armas_explosivos,
                    "sentimiento": sentiment,
                    "class_label": class_label
                }
                alerts.append(alert)

                # Guardar en base de datos
                try:
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO alertas (oracion, tipo_novedad, ubicacion, identidades, fechas, vehiculo, armas_explosivos, sentimiento, class_label, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        sent, alert["tipo_novedad"], ubicacion, identidades, fechas, vehiculo_str, armas_explosivos,
                        sentiment, class_label, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    conn.commit()
                    print(f"Alerta guardada en la base de datos: {sent}")
                except Exception as e:
                    st.error(f"Error al guardar en la base de datos: {e}")
                    print(f"Error al guardar en la base de datos: {e}")
                finally:
                    conn.close()

        # Guardar resultados en el estado
        st.session_state.alerts = alerts

        # Mostrar resultados
        st.markdown("### Análisis de Entidades")
        st.markdown(highlight_entities(text_clean), unsafe_allow_html=True)

        st.markdown("### Análisis de Sentimiento")
        st.markdown(sentiment_html, unsafe_allow_html=True)

        st.markdown("### Análisis de Prioridad")
        st.dataframe(tabla_prioridades)
        # Resumen de prioridades
        resumen_prioridades = pd.value_counts([d["Clase de Prioridad"] for d in prioridades_data])
        st.markdown("**Resumen de Prioridades**")
        for prioridad, count in resumen_prioridades.items():
            st.write(f"{prioridad}: {count} oración(es)")

        st.markdown("### Identificación de Novedades")
        if alerts:
            st.table(pd.DataFrame(alerts))
            st.session_state.pdf_buffer = generar_reporte_pdf(alerts)
            st.success("Análisis completado. Puedes descargar el PDF.")
        else:
            st.write("No se detectaron novedades que cumplan los criterios.")

    # Botón para descargar el PDF
    if st.button("Generar Reporte PDF", key="generate_pdf_button"):
        if st.session_state.alerts:
            pdf_filename = pdf_filename_input.strip() + ".pdf" if pdf_filename_input.strip() else "reporte_novedades.pdf"
            if st.session_state.pdf_buffer is None:
                st.session_state.pdf_buffer = generar_reporte_pdf(st.session_state.alerts)
            st.download_button(
                label="Descargar Reporte PDF",
                data=st.session_state.pdf_buffer,
                file_name=pdf_filename,
                mime="application/pdf",
                key="download_pdf_button"
            )
        else:
            st.warning("No hay alertas para generar el reporte. Analiza un texto primero.")

if __name__ == "__main__":
    main()