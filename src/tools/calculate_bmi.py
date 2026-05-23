"""BMI calculator tool: computes Body Mass Index and returns a WHO category in German.

Used by the agent when a clinical query involves patient weight/height context
(e.g. eligibility assessment, dosing by body surface area).
"""


def calculate_bmi_tool(weight_kg: float, height_cm: float) -> dict:
    """Calculate BMI and return a German WHO-style category."""
    if weight_kg <= 0:
        return {"error": "weight_kg muss größer als 0 sein."}
    if height_cm <= 0:
        return {"error": "height_cm muss größer als 0 sein."}

    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)

    # WHO BMI classification thresholds (same in German-language clinical guidelines).
    if bmi < 18.5:
        category = "Untergewicht"
    elif bmi < 25:
        category = "Normalgewicht"
    elif bmi < 30:
        category = "Übergewicht"
    else:
        category = "Adipositas"

    return {
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": round(bmi, 2),
        "category": category,
    }
