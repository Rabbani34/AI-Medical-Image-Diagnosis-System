"""
chatbot.py — AI Doctor Chatbot
================================
Primary:  Anthropic Claude API  (requires ANTHROPIC_API_KEY env var)
Fallback: Rule-based responses  (works without API key)

Usage:
    from chatbot import medical_chatbot
    reply = medical_chatbot("What medicine should I take?", disease="PNEUMONIA")
"""

import os

# ── Claude system prompt ───────────────────────────────────────────────────────
_SYSTEM = """You are a compassionate, knowledgeable AI medical assistant embedded in a
lung-disease diagnosis platform. You help patients understand their AI-generated
chest X-ray diagnosis results (Normal, Pneumonia, or Tuberculosis).

Guidelines:
• Be clear, empathetic, and concise (2–4 sentences per reply).
• Provide accurate, evidence-based information about lung conditions.
• Tailor your answer to the patient's current diagnosis when it is provided.
• Always remind the patient to consult a licensed medical professional for
  actual diagnosis, treatment decisions, or prescription medications.
• Do NOT diagnose, prescribe medication doses, or replace a real doctor.
• If the question is unrelated to medical / lung health, politely redirect.
"""


def medical_chatbot(user_input: str, disease: str | None = None) -> str:
    """
    Return a helpful AI medical response.

    Parameters
    ----------
    user_input : str   — patient's question
    disease    : str | None — "NORMAL" | "PNEUMONIA" | "TUBERCULOSIS" | None

    Returns
    -------
    str — response text
    """
    # ── Try Claude first ──────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            return _claude_response(user_input, disease, api_key)
        except Exception as exc:
            print(f"[chatbot] Claude API error ({exc}), falling back to rule-based.")

    # ── Fallback ──────────────────────────────────────────────────────────────
    return _rule_based(user_input, disease)


# ── Claude response ────────────────────────────────────────────────────────────

def _claude_response(user_input: str, disease: str | None, api_key: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    context = (
        f"The patient's current AI diagnosis result is: **{disease}**."
        if disease else
        "No diagnosis has been performed yet for this patient."
    )

    user_msg = f"{context}\n\nPatient's question: {user_input}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based(user_input: str, disease: str | None) -> str:
    q = user_input.lower()

    # Greeting
    if any(w in q for w in ["hi", "hello", "hey"]):
        return (
            "Hello! I'm your AI medical assistant. "
            "Ask me anything about your diagnosis, symptoms, medication, or lung health."
        )

    # Medicine / treatment
    if any(w in q for w in ["medicine", "treatment", "medication", "drug", "antibiotic"]):
        if disease == "PNEUMONIA":
            return (
                "Pneumonia is typically treated with antibiotics such as amoxicillin or "
                "azithromycin, combined with rest and adequate hydration. "
                "Always follow your doctor's prescription."
            )
        if disease == "TUBERCULOSIS":
            return (
                "TB treatment involves a long course of antibiotics — isoniazid, rifampicin, "
                "pyrazinamide, and ethambutol — under DOTS (Directly Observed Therapy). "
                "Never skip doses; completing the full course is critical."
            )
        return (
            "Medication depends on the specific diagnosis and your doctor's evaluation. "
            "Please consult a licensed medical professional before taking any medicine."
        )

    # Diet / food
    if any(w in q for w in ["diet", "food", "eat", "nutrition", "drink"]):
        if disease == "TUBERCULOSIS":
            return (
                "For TB, a high-protein diet (eggs, milk, legumes) along with vitamin-rich "
                "fruits and vegetables supports recovery. Stay well hydrated and avoid junk food."
            )
        if disease == "PNEUMONIA":
            return (
                "Warm fluids like soups and herbal teas help with pneumonia recovery. "
                "Include high-protein foods (eggs, pulses) and avoid cold food and drinks."
            )
        return (
            "A lung-healthy diet includes fresh fruits, vegetables, lean proteins, and plenty "
            "of water. Avoid smoking, alcohol, and highly processed foods."
        )

    # Exercise / activity
    if any(w in q for w in ["exercise", "activity", "workout", "walk", "yoga"]):
        if disease in ("PNEUMONIA", "TUBERCULOSIS"):
            return (
                "While recovering, stick to light activity — gentle breathing exercises and "
                "short walks. Avoid strenuous workouts until your doctor clears you."
            )
        return (
            "Regular breathing exercises, walking, and yoga can strengthen lung capacity. "
            "Aim for at least 20–30 minutes of moderate activity most days."
        )

    # Symptoms
    if any(w in q for w in ["symptom", "sign", "feel", "cough", "fever", "breath"]):
        return (
            "Common lung disease symptoms include persistent cough, fever, chest pain, "
            "shortness of breath, and fatigue. "
            "Seek medical attention promptly if symptoms worsen."
        )

    # Disease-specific info
    if "pneumonia" in q:
        return (
            "Pneumonia is a lung infection causing inflammation of the air sacs, often due "
            "to bacteria, viruses, or fungi. It can range from mild to life-threatening."
        )
    if "tuberculosis" in q or " tb " in q or q.endswith("tb"):
        return (
            "Tuberculosis (TB) is a contagious bacterial infection caused by "
            "Mycobacterium tuberculosis, mainly affecting the lungs. "
            "It is treatable with a full course of antibiotics."
        )

    # Condition-based default
    if disease == "PNEUMONIA":
        return (
            "You've been diagnosed with Pneumonia. Follow your prescribed antibiotics, get "
            "plenty of rest, stay warm, and consult your doctor regularly."
        )
    if disease == "TUBERCULOSIS":
        return (
            "TB requires strict adherence to DOTS therapy. Never skip doses, wear a mask to "
            "prevent spreading, and attend all follow-up appointments."
        )
    if disease == "NORMAL":
        return (
            "Your X-ray appears normal — no significant abnormalities detected. "
            "Maintain a healthy lifestyle and schedule routine check-ups."
        )

    return (
        "I'm here to help with questions about your lung health and diagnosis. "
        "For specific medical advice, always consult a qualified healthcare professional."
    )