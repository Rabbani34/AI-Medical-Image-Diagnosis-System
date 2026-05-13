def generate_report(disease, confidence, severity):

    if disease == "NORMAL":

        return f"""
Radiology Report
-------------------------------

No abnormal lung findings detected.

Confidence Level: {confidence:.2f}%

Severity: None

Recommendation:
Routine monitoring only.
"""


    if disease == "PNEUMONIA":

        return f"""
Radiology Report
-------------------------------

Patchy lung opacities detected.

Possible pneumonia infection.

Confidence Level: {confidence:.2f}%

Severity: {severity}

Recommendation:
Clinical evaluation recommended.
Antibiotic therapy may be required.
"""


    if disease == "TUBERCULOSIS":

        return f"""
Radiology Report
-------------------------------

Suspicious cavitary lesions detected in lung region.

Findings suggest tuberculosis infection.

Confidence Level: {confidence:.2f}%

Severity: {severity}

Recommendation:
Immediate TB diagnostic testing required.
Isolation precautions advised.
"""