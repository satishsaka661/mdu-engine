"""
MDU Engine — Mailer
Sends OTP login codes via Gmail SMTP.
Uses your existing satish@mduengine.com Gmail account.

Environment variables required:
    MDU_EMAIL_SENDER    — sender address e.g. satish@mduengine.com
    MDU_EMAIL_PASSWORD  — Gmail App Password (NOT your Gmail password)

How to get Gmail App Password:
1. Go to myaccount.google.com
2. Security → 2-Step Verification → App passwords
3. Create one for "MDU Engine"
4. Copy the 16-character password
5. Set as MDU_EMAIL_PASSWORD in Cloud Run environment variables
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


SENDER_EMAIL   = os.environ.get("MDU_EMAIL_SENDER", "satish@mduengine.com")
SENDER_PASSWORD = os.environ.get("MDU_EMAIL_PASSWORD", "")
SMTP_HOST      = "smtp.zoho.in"
SMTP_PORT      = 587


def send_otp_email(recipient_email: str, name: str, otp: str) -> dict:
    """
    Send a 6-digit OTP to recipient_email.
    Returns {"success": bool, "error": str}
    """
    if not SENDER_PASSWORD:
        return {
            "success": False,
            "error": "Email not configured. Set MDU_EMAIL_PASSWORD environment variable."
        }

    subject = "Your MDU Engine login code"

    html_body = f"""
    <div style="font-family: Inter, sans-serif; max-width: 480px; margin: 0 auto; 
                background: #0D1B2A; color: #E8F0FE; padding: 32px; border-radius: 12px;">

        <div style="font-size: 22px; font-weight: 800; 
                    background: linear-gradient(90deg, #1E90FF 0%, #FF6B2B 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin-bottom: 8px;">
            MDU Engine
        </div>

        <div style="color: #FF6B2B; font-size: 12px; font-weight: 600; 
                    letter-spacing: 0.1em; margin-bottom: 24px;">
            DECISION-SUPPORT PLATFORM
        </div>

        <p style="color: #E8F0FE; font-size: 15px; margin-bottom: 8px;">
            Hi {name},
        </p>

        <p style="color: #8BA3C7; font-size: 14px; margin-bottom: 24px;">
            Here is your login code for MDU Engine.
        </p>

        <div style="background: #1E3A5F; border: 1px solid #1E90FF; border-radius: 10px; 
                    padding: 24px; text-align: center; margin-bottom: 24px;">
            <div style="color: #8BA3C7; font-size: 12px; 
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">
                Your login code
            </div>
            <div style="color: #1E90FF; font-size: 42px; font-weight: 800; 
                        letter-spacing: 0.2em;">
                {otp}
            </div>
            <div style="color: #8BA3C7; font-size: 12px; margin-top: 8px;">
                Expires in 10 minutes
            </div>
        </div>

        <p style="color: #5A7A9F; font-size: 12px; margin-bottom: 4px;">
            If you didn't request this, ignore this email.
        </p>
        <p style="color: #5A7A9F; font-size: 12px;">
            MDU Engine · app.mduengine.com · satish@mduengine.com
        </p>
    </div>
    """

    plain_body = f"""
Hi {name},

Your MDU Engine login code is: {otp}

This code expires in 10 minutes.

If you didn't request this, ignore this email.

MDU Engine · app.mduengine.com
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"MDU Engine <{SENDER_EMAIL}>"
        msg["To"]      = recipient_email

        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())

        return {"success": True, "error": ""}

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "Email authentication failed. Check MDU_EMAIL_PASSWORD is a valid Gmail App Password."
        }
    except Exception as e:
        return {"success": False, "error": f"Email send failed: {str(e)}"}