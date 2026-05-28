import os
import webbrowser


def generate_printable_label(patient, qr_path):

    html = f"""
    <html>
    <head>
        <title>Patient QR Label</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                padding-top: 50px;
            }}

            img {{
                width: 250px;
                height: 250px;
            }}

            .name {{
                font-size: 28px;
                font-weight: bold;
                margin-top: 20px;
            }}

            .uuid {{
                font-size: 20px;
                margin-top: 10px;
            }}

            @media print {{
                body {{
                    margin: 0;
                }}
            }}
        </style>
    </head>

    <body onload="window.print()">

        <img src="{qr_path}" />

        <div class="name">
            {patient["first_name"]} {patient["last_name"]}
        </div>

        <div class="uuid">
            UUID: {patient["patient_uuid"]}
        </div>

    </body>
    </html>
    """

    save_path = "temp_patient_label.html"

    with open(save_path, "w") as f:
        f.write(html)

    webbrowser.open("file://" + os.path.abspath(save_path))