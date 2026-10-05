"""
app.py
-------
Flask application for the AI-Powered Healthcare Diagnosis Assistant.

Routes:
  GET  /                 -> symptom intake form
  POST /predict           -> runs the pipeline, saves to history, shows report
  POST /feedback/<id>     -> save "was this helpful?" feedback for a report
  GET  /history            -> table of past checks (current user's, if logged in)
  GET  /history/<id>       -> re-view a past report
  POST /history/clear      -> wipe history
  GET  /history/export     -> download history as CSV
  GET  /dashboard           -> analytics dashboard (charts)
  GET  /api/dashboard-data -> JSON data for the dashboard charts
  GET  /health-tips         -> seasonal disease awareness (India-focused) page
  GET  /find-doctor          -> sample specialist directory (demo data)
  GET  /emergency-hospitals   -> live-location nearest hospital finder (map)
  GET  /api/nearby-hospitals -> JSON: hospitals near a given lat/lng
  GET  /api/nearby-specialists -> JSON: doctors/clinics near a lat/lng matching a specialty
  GET/POST /register, /login, GET /logout -> simple account system

Run:  python app.py   then open http://localhost:5000
"""

import csv
import io
import functools
import os

import requests
from flask import (
    Flask, render_template, request, redirect, url_for, flash, session,
    jsonify, Response,
)

import database
import predict

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

# Optional live doctor search via the Google Places API. If this isn't set,
# /find-doctor gracefully falls back to the curated static listings below —
# the feature works either way. See README for how to get a free API key.
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()

database.init_db()


# ------------------------------------------------------------- helpers ----

def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def current_user():
    user_id = session.get("user_id")
    return database.get_user_by_id(user_id) if user_id else None


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to view this page.")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    return {"logged_in_user": current_user()}


# --------------------------------------------------------------- auth ----

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()

        if not username or not password:
            flash("Username and password are required.")
            return redirect(url_for("register"))
        if len(password) < 4:
            flash("Password must be at least 4 characters.")
            return redirect(url_for("register"))

        user_id = database.create_user(username, password, full_name)
        if user_id is None:
            flash("That username is already taken.")
            return redirect(url_for("register"))

        session["user_id"] = user_id
        flash(f"Welcome, {full_name or username}! Your account was created.")
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = database.get_user_by_username(username)

        if user is None or not database.verify_password(user, password):
            flash("Invalid username or password.")
            return redirect(url_for("login"))

        session["user_id"] = user["id"]
        flash(f"Welcome back, {user['full_name'] or user['username']}!")
        next_url = request.args.get("next") or url_for("index")
        return redirect(next_url)

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("You have been logged out.")
    return redirect(url_for("index"))


# ---------------------------------------------------------- core flow ----

@app.route("/")
def index():
    symptom_list = predict.load_symptom_list()  # [(key, display_name, severity), ...]
    return render_template("index.html", symptoms=symptom_list)


@app.route("/predict", methods=["POST"])
def predict_route():
    form = request.form

    patient = {
        "age": _parse_int(form.get("age")),
        "gender": form.get("gender", "Not specified"),
        "weight_kg": _parse_float(form.get("weight_kg")),
        "height_cm": _parse_float(form.get("height_cm")),
        "temperature_c": _parse_float(form.get("temperature_c")),
        "duration_days": _parse_int(form.get("duration_days")),
        "existing_conditions": form.getlist("existing_conditions"),
        "symptoms": form.getlist("symptoms"),
    }

    if not patient["symptoms"]:
        flash("Please select at least one symptom before submitting.")
        return redirect(url_for("index"))

    result = predict.run_pipeline(patient)
    user = current_user()
    record_id = database.save_prediction(patient, result, user_id=user["id"] if user else None)

    return render_template(
        "prediction.html",
        patient=patient,
        result=result,
        record_id=record_id,
    )


@app.route("/feedback/<int:record_id>", methods=["POST"])
def feedback_route(record_id):
    is_helpful = request.form.get("is_helpful") == "yes"
    comment = request.form.get("comment", "").strip() or None
    database.save_feedback(record_id, is_helpful, comment)
    flash("Thanks for your feedback — it helps improve this project!")
    return redirect(url_for("history_detail", record_id=record_id))


# ------------------------------------------------------------ history ----

@app.route("/history")
def history():
    user = current_user()
    records = database.get_history(user_id=user["id"] if user else None)
    return render_template("history.html", records=records, user=user)


@app.route("/history/<int:record_id>")
def history_detail(record_id):
    record = database.get_prediction_by_id(record_id)
    if record is None:
        flash("That record could not be found.")
        return redirect(url_for("history"))

    recommendations = predict.get_recommendations(record["top_disease"]) if record["top_disease"] else {
        "precautions": ["No prediction was generated for this record."],
        "medication_info": {"otc": [], "avoid": [], "prescription_note": "N/A", "urgent": False},
        "specialist": "General Physician",
    }
    result = {
        "bmi": record["bmi"],
        "bmi_category": record["bmi_category"],
        "emergency": {"is_emergency": bool(record["is_emergency"]), "triggered_by": []},
        "predictions": [(record["top_disease"], record["confidence"])] if record["top_disease"] else [],
        "risk": {"level": record["risk_level"], "score": None},
        "recommendations": recommendations,
        "top_disease": record["top_disease"],
    }
    patient = {
        "age": record["age"],
        "gender": record["gender"],
        "weight_kg": record["weight_kg"],
        "height_cm": record["height_cm"],
        "temperature_c": record["temperature_c"],
        "duration_days": record["duration_days"],
        "existing_conditions": record["existing_conditions"],
        "symptoms": record["symptoms"],
    }
    return render_template(
        "prediction.html", patient=patient, result=result, record_id=record_id, from_history=True
    )


@app.route("/history/clear", methods=["POST"])
def history_clear():
    user = current_user()
    database.clear_history(user_id=user["id"] if user else None)
    flash("History cleared.")
    return redirect(url_for("history"))


@app.route("/history/export")
def history_export():
    user = current_user()
    records = database.get_history(user_id=user["id"] if user else None)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "Date", "Age", "Gender", "BMI", "BMI Category", "Temperature (C)",
        "Duration (days)", "Existing Conditions", "Symptoms",
        "Top Predicted Condition", "Confidence (%)", "Risk Level",
        "Emergency Flag", "Recommended Specialist",
    ])
    for r in records:
        writer.writerow([
            r["created_at"], r["age"], r["gender"], r["bmi"], r["bmi_category"],
            r["temperature_c"], r["duration_days"],
            "; ".join(r["existing_conditions"]), "; ".join(r["symptoms"]),
            r["top_disease"], r["confidence"], r["risk_level"],
            "Yes" if r["is_emergency"] else "No", r["specialist"],
        ])

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=diagnosis_history.csv"},
    )


# ----------------------------------------------------------- dashboard ----

@app.route("/dashboard")
@login_required
def dashboard():
    metrics = predict.load_model_metrics()
    return render_template("dashboard.html", metrics=metrics)


@app.route("/api/dashboard-data")
@login_required
def api_dashboard_data():
    user = current_user()
    stats = database.get_dashboard_stats(user_id=user["id"])
    stats["feedback"] = database.get_feedback_stats()
    return jsonify(stats)


# ---------------------------------------------------- extra info pages ----

SEASONAL_HEALTH_TIPS = [
    {
        "season": "Monsoon (Jun–Sep)",
        "watch_for": ["Dengue Fever", "Malaria", "Chikungunya", "Cholera", "Typhoid Fever"],
        "tips": [
            "Eliminate stagnant water around your home to prevent mosquito breeding",
            "Drink only boiled or purified water",
            "Use mosquito repellents and nets, especially at dusk and dawn",
            "Avoid street food during heavy rain periods",
        ],
    },
    {
        "season": "Winter (Nov–Feb)",
        "watch_for": ["Common Cold", "Influenza (Flu)", "Bronchitis", "Pneumonia", "Asthma"],
        "tips": [
            "Keep warm, especially in the early morning and at night",
            "Get a seasonal flu vaccine if advised by your doctor",
            "Ventilate closed rooms to reduce indoor respiratory infections",
            "Elderly and asthma patients should avoid heavy smog exposure",
        ],
    },
    {
        "season": "Summer (Mar–Jun)",
        "watch_for": ["Gastroenteritis", "Typhoid Fever", "Hepatitis A", "Heat-related illness"],
        "tips": [
            "Stay hydrated and avoid prolonged sun exposure",
            "Avoid unhygienic street food and unpurified water",
            "Wash fruits and vegetables thoroughly before eating",
            "Watch for signs of dehydration in children and the elderly",
        ],
    },
]


@app.route("/health-tips")
def health_tips():
    return render_template("health_tips.html", seasons=SEASONAL_HEALTH_TIPS)


# Real specialist / hospital listings for Agra, Mathura & nearby areas,
# sourced from public Google Maps listings (name, address, phone, rating as
# publicly shown). Curated for strong ratings and genuine review volume.
# IMPORTANT: verify details (hours, current doctor, fees) by calling ahead —
# listings can change and this is not a live, continuously-updated feed.
SAMPLE_CLINICS = [
    {
        "name": "Aakash Healthcare Agra – Multi Speciality Hospital",
        "specialists": "General Physician, Emergency Care, Multi-specialty",
        "city": "Agra",
        "address": "672, Mathura Bypass Road, NH-2, Sikandra, Agra, UP 282007",
        "phone": "+91 562 613 8888",
        "rating": "5.0 (1180+ reviews)",
    },
    {
        "name": "Yatharth Super Speciality Hospital, Agra",
        "specialists": "Cardiologist, Neurologist, Gynecologist, Multi-specialty",
        "city": "Agra",
        "address": "610, Kakretha Road, NH 19, near Guru ka Taal, Agra, UP 282007",
        "phone": "+91 79002 22220",
        "rating": "4.7 (1750+ reviews)",
    },
    {
        "name": "Global Heart Institute — Dr. Suvir Gupta",
        "specialists": "Cardiologist / Angina, Heart Disease",
        "city": "Agra",
        "address": "Shanti Madhuban Plaza, Delhi Gate Rd, Civil Lines, Agra, UP 282002",
        "phone": "+91 73009 11117",
        "rating": "4.8 (1600+ reviews)",
    },
    {
        "name": "Dr. D P Sharma — Bone Hospital",
        "specialists": "Orthopedist (joint replacement, ACL, fractures)",
        "city": "Agra",
        "address": "Bone Hospital, Adarsh Nagar, Pratap Pura, Rakabganj, Agra, UP 282001",
        "phone": "+91 81287 76467",
        "rating": "4.9 (260+ reviews)",
    },
    {
        "name": "Dr. L K Gupta — ENT Specialist",
        "specialists": "ENT Specialist (Sinusitis, Tonsillitis, Ear Infections)",
        "city": "Agra",
        "address": "R P Medicare, 51/1 Paliwal Park Rd, Gandhi Nagar, Civil Lines, Agra, UP 282003",
        "phone": "+91 89585 67070",
        "rating": "4.9 (250+ reviews)",
    },
    {
        "name": "Dr. Smita Raghav — Shakti Healthcare Hospital",
        "specialists": "Gynecologist / PCOS / Anemia in pregnancy",
        "city": "Agra",
        "address": "Shakti Healthcare Hospital, Gailana Rd, RK Puram, Agra, UP 282007",
        "phone": "+91 78954 81436",
        "rating": "4.9 (120+ reviews)",
    },
    {
        "name": "APDERM — Dr. Param Satsangi / Dr. Ampi Padi",
        "specialists": "Dermatologist (Fungal Skin Infection, Acne)",
        "city": "Agra",
        "address": "28, Dev Nagar, Khandari, Agra, UP 282005",
        "phone": "+91 70781 49198",
        "rating": "4.9 (110+ reviews)",
    },
    {
        "name": "Sarthak Nursing Home",
        "specialists": "General Surgeon, Urologist (Kidney Stones), Gynecologist",
        "city": "Agra",
        "address": "G-23, Bypass Rd, Ghatwasan, Kamla Nagar, Agra, UP 282005",
        "phone": "+91 98375 23666",
        "rating": "4.9 (900+ reviews)",
    },
    {
        "name": "Sarvodaya Hospital, Mathura",
        "specialists": "Multi-specialty, Physiotherapy, General Physician",
        "city": "Mathura",
        "address": "NH19, Mathura, UP 281003",
        "phone": "+91 96546 11414",
        "rating": "4.7 (170+ reviews)",
    },
    {
        "name": "Aakash CIMS, Mathura",
        "specialists": "Multi-specialty, Oncology, General Physician",
        "city": "Mathura",
        "address": "National Highway 19, Masani Rd, Mathura, UP 281001",
        "phone": "+91 92581 13570",
        "rating": "4.6 (2000+ reviews)",
    },
    {
        "name": "Bhaskar Super Speciality Hospital — Dr. Bhaskar Tiwari",
        "specialists": "Orthopedist (Joint & Hip Surgery)",
        "city": "Mathura",
        "address": "A-6, NH 19, Madhavpuri, Mathura, UP 281001",
        "phone": "+91 90847 01930",
        "rating": "4.6 (630+ reviews)",
    },
    {
        "name": "Dr. Love Gupta — Respiratory & Chest Specialist",
        "specialists": "Pulmonologist (Asthma, TB, COPD, Bronchitis)",
        "city": "Mathura",
        "address": "CIMS, NH 19, near Radha Valley, Mathura, UP 281001",
        "phone": "+91 92581 13570",
        "rating": "4.9 (95+ reviews)",
    },
    {
        "name": "Dr. Shradha Soni",
        "specialists": "Pulmonologist / Allergist (Asthma, Allergic Rhinitis)",
        "city": "Mathura",
        "address": "Shop 51, Shri Square Mall, near Radha Puram, Mathura, UP 281001",
        "phone": "+91 84455 62272",
        "rating": "4.8 (330+ reviews)",
    },
    {
        "name": "Shri Om Hospital — Dr. Mansi Agrawal",
        "specialists": "Gynecologist",
        "city": "Mathura",
        "address": "Bhuteshwar Rd, Manoharpura, Mathura, UP 281001",
        "phone": "+91 97205 41461",
        "rating": "4.6 (250+ reviews)",
    },
    {
        "name": "Placenta Mother Child Multispeciality — Dr. Om Chaudhary",
        "specialists": "Pediatrician",
        "city": "Mathura",
        "address": "Township Chauraha, near Swarna Jayanti Hospital, Mathura, UP 281006",
        "phone": "+91 94587 06888",
        "rating": "4.9 (95+ reviews)",
    },
]


@app.route("/find-doctor")
def find_doctor():
    specialist_query = request.args.get("specialist", "").strip()
    city = request.args.get("city", "").strip()
    searched = request.args.get("searched") == "1"
    auto_locate = request.args.get("auto_locate") == "1"

    live_results = None
    live_error = None

    if searched:
        if not city:
            live_error = "Please type a city/area name, or use the live location button above instead."
        elif GOOGLE_PLACES_API_KEY:
            live_results = search_doctors_live(specialist_query, city)
            if live_results == []:
                live_error = "No live results found for that search — showing curated listings instead."
        else:
            live_error = (
                "Live search isn't configured on this deployment yet — showing curated listings instead. "
                "See the README for how to enable it with a free Google Places API key."
            )

    return render_template(
        "doctors.html",
        clinics=SAMPLE_CLINICS,
        specialist_query=specialist_query,
        city=city,
        searched=searched,
        live_results=live_results,
        live_error=live_error,
        live_enabled=bool(GOOGLE_PLACES_API_KEY),
        auto_locate=auto_locate,
    )


def search_doctors_live(specialty, city):
    """
    Live search via the Google Places API Text Search endpoint.
    Returns a list of {name, address, rating, user_ratings_total, maps_url}
    dicts, or [] if the API call fails or finds nothing.
    """
    query_text = f"{specialty} doctor hospital in {city}" if specialty else f"hospitals in {city}"
    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": query_text, "key": GOOGLE_PLACES_API_KEY},
            timeout=6,
        )
        data = response.json()
        if data.get("status") != "OK":
            return []

        results = []
        for item in data.get("results", [])[:12]:
            place_id = item.get("place_id")
            results.append({
                "name": item.get("name"),
                "address": item.get("formatted_address"),
                "rating": item.get("rating"),
                "user_ratings_total": item.get("user_ratings_total"),
                "maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else None,
            })
        return results
    except requests.RequestException:
        return []


# ------------------------------------------------- emergency hospital finder ----

INDIA_EMERGENCY_NUMBERS = [
    {"label": "National Emergency Number (Police/Fire/Medical)", "number": "112"},
    {"label": "Ambulance", "number": "108"},
]


@app.route("/emergency-hospitals")
def emergency_hospitals():
    return render_template(
        "emergency_map.html",
        emergency_numbers=INDIA_EMERGENCY_NUMBERS,
        live_enabled=bool(GOOGLE_PLACES_API_KEY),
    )


def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2) ** 2
    return round(r * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def _nearby_hospitals_google(lat, lng, radius_m=8000):
    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{lat},{lng}",
                "radius": radius_m,
                "type": "hospital",
                "key": GOOGLE_PLACES_API_KEY,
            },
            timeout=6,
        )
        data = response.json()
        if data.get("status") != "OK":
            return None  # signal caller to fall back

        results = []
        for item in data.get("results", [])[:15]:
            loc = item.get("geometry", {}).get("location", {})
            h_lat, h_lng = loc.get("lat"), loc.get("lng")
            if h_lat is None or h_lng is None:
                continue
            place_id = item.get("place_id")
            results.append({
                "name": item.get("name"),
                "address": item.get("vicinity"),
                "lat": h_lat,
                "lng": h_lng,
                "rating": item.get("rating"),
                "distance_km": _haversine_km(lat, lng, h_lat, h_lng),
                "maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else None,
                "directions_url": f"https://www.google.com/maps/dir/?api=1&origin={lat},{lng}&destination={h_lat},{h_lng}",
            })
        results.sort(key=lambda r: r["distance_km"])
        return results
    except requests.RequestException:
        return None


# Public Overpass mirrors, tried in order — the main server can be slow or
# briefly unavailable, so we fall back to alternates rather than failing.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def _query_overpass(query):
    """Tries each Overpass mirror in turn; returns parsed JSON or raises."""
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"[overpass] trying {endpoint} ...")
            response = requests.post(endpoint, data={"data": query}, timeout=25)
            response.raise_for_status()
            data = response.json()
            print(f"[overpass] {endpoint} OK, {len(data.get('elements', []))} elements")
            return data
        except Exception as exc:  # noqa: BLE001 - we want to try the next mirror on ANY failure
            print(f"[overpass] {endpoint} FAILED: {exc}")
            last_error = exc
    raise RuntimeError(f"All Overpass mirrors failed: {last_error}")


def _nearby_hospitals_osm(lat, lng, radius_m=12000):
    """Free fallback using OpenStreetMap's Overpass API — no API key needed."""
    query = f"""
    [out:json][timeout:20];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lng});
      way["amenity"="hospital"](around:{radius_m},{lat},{lng});
      node["healthcare"="hospital"](around:{radius_m},{lat},{lng});
    );
    out center 20;
    """
    try:
        data = _query_overpass(query)
        results = []
        for el in data.get("elements", []):
            if el.get("type") == "way":
                center = el.get("center", {})
                h_lat, h_lng = center.get("lat"), center.get("lon")
            else:
                h_lat, h_lng = el.get("lat"), el.get("lon")
            if h_lat is None or h_lng is None:
                continue
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue  # skip unnamed nodes, not useful to show
            address_parts = [
                tags.get("addr:housenumber"), tags.get("addr:street"),
                tags.get("addr:city"),
            ]
            address = ", ".join(p for p in address_parts if p) or None
            results.append({
                "name": name,
                "address": address,
                "lat": h_lat,
                "lng": h_lng,
                "rating": None,
                "distance_km": _haversine_km(lat, lng, h_lat, h_lng),
                "maps_url": f"https://www.openstreetmap.org/?mlat={h_lat}&mlon={h_lng}#map=18/{h_lat}/{h_lng}",
                "directions_url": f"https://www.google.com/maps/dir/?api=1&origin={lat},{lng}&destination={h_lat},{h_lng}",
            })
        results.sort(key=lambda r: r["distance_km"])
        return results[:15]
    except Exception as exc:  # noqa: BLE001
        print(f"[nearby_hospitals_osm] failed: {exc}")
        return []


@app.route("/api/nearby-hospitals")
def api_nearby_hospitals():
    lat = _parse_float(request.args.get("lat"))
    lng = _parse_float(request.args.get("lng"))
    if lat is None or lng is None:
        return jsonify({"error": "lat and lng query parameters are required"}), 400

    source = "openstreetmap"
    results = None
    if GOOGLE_PLACES_API_KEY:
        results = _nearby_hospitals_google(lat, lng)
        if results is not None:
            source = "google"

    if results is None:
        results = _nearby_hospitals_osm(lat, lng)

    return jsonify({"source": source, "count": len(results), "hospitals": results})


def _nearby_specialists_google(lat, lng, specialty, radius_m=10000):
    try:
        params = {
            "location": f"{lat},{lng}",
            "radius": radius_m,
            "type": "doctor",
            "key": GOOGLE_PLACES_API_KEY,
        }
        if specialty:
            params["keyword"] = specialty
        response = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params=params,
            timeout=6,
        )
        data = response.json()
        if data.get("status") != "OK":
            return None

        results = []
        for item in data.get("results", [])[:15]:
            loc = item.get("geometry", {}).get("location", {})
            h_lat, h_lng = loc.get("lat"), loc.get("lng")
            if h_lat is None or h_lng is None:
                continue
            place_id = item.get("place_id")
            results.append({
                "name": item.get("name"),
                "address": item.get("vicinity"),
                "lat": h_lat,
                "lng": h_lng,
                "rating": item.get("rating"),
                "distance_km": _haversine_km(lat, lng, h_lat, h_lng),
                "maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else None,
                "directions_url": f"https://www.google.com/maps/dir/?api=1&origin={lat},{lng}&destination={h_lat},{h_lng}",
            })
        results.sort(key=lambda r: r["distance_km"])
        return results
    except requests.RequestException:
        return None


def _nearby_specialists_osm(lat, lng, specialty, radius_m=12000):
    """
    Free fallback via OpenStreetMap's Overpass API. OSM's specialty tagging is
    sparse, so this does a loose name/tag match for the given specialty and
    falls back to returning all nearby clinics/doctors if nothing matches —
    better to show something nearby than nothing at all.
    """
    query = f"""
    [out:json][timeout:20];
    (
      node["amenity"="doctors"](around:{radius_m},{lat},{lng});
      node["amenity"="clinic"](around:{radius_m},{lat},{lng});
      way["amenity"="clinic"](around:{radius_m},{lat},{lng});
      node["amenity"="hospital"](around:{radius_m},{lat},{lng});
      way["amenity"="hospital"](around:{radius_m},{lat},{lng});
    );
    out center 30;
    """
    try:
        data = _query_overpass(query)
        results = []
        specialty_lower = (specialty or "").lower()
        for el in data.get("elements", []):
            if el.get("type") == "way":
                center = el.get("center", {})
                h_lat, h_lng = center.get("lat"), center.get("lon")
            else:
                h_lat, h_lng = el.get("lat"), el.get("lon")
            if h_lat is None or h_lng is None:
                continue
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue
            haystack = f"{name} {tags.get('healthcare:speciality', '')}".lower()
            matched = bool(specialty_lower) and specialty_lower in haystack
            address_parts = [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")]
            address = ", ".join(p for p in address_parts if p) or None
            results.append({
                "name": name,
                "address": address,
                "lat": h_lat,
                "lng": h_lng,
                "rating": None,
                "matched_specialty": matched,
                "distance_km": _haversine_km(lat, lng, h_lat, h_lng),
                "maps_url": f"https://www.openstreetmap.org/?mlat={h_lat}&mlon={h_lng}#map=18/{h_lat}/{h_lng}",
                "directions_url": f"https://www.google.com/maps/dir/?api=1&origin={lat},{lng}&destination={h_lat},{h_lng}",
            })
        # Show specialty-matching results first (if any), then nearest overall.
        results.sort(key=lambda r: (not r["matched_specialty"], r["distance_km"]))
        return results[:20]
    except Exception as exc:  # noqa: BLE001
        print(f"[nearby_specialists_osm] failed: {exc}")
        return []


@app.route("/api/nearby-specialists")
def api_nearby_specialists():
    """
    Live location + condition-aware doctor search: finds nearby doctors/clinics
    matching the recommended specialty for the person's predicted condition.
    """
    lat = _parse_float(request.args.get("lat"))
    lng = _parse_float(request.args.get("lng"))
    specialty = request.args.get("specialty", "").strip()
    if lat is None or lng is None:
        return jsonify({"error": "lat and lng query parameters are required"}), 400

    source = "openstreetmap"
    results = None
    if GOOGLE_PLACES_API_KEY:
        results = _nearby_specialists_google(lat, lng, specialty)
        if results is not None:
            source = "google"

    if results is None:
        results = _nearby_specialists_osm(lat, lng, specialty)

    return jsonify({"source": source, "specialty": specialty, "count": len(results), "specialists": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
