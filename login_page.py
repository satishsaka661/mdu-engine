"""
login_page.py — MDU Engine Login UI
Magic link OTP login. No password required.
Call show_login_gate() at the top of app.py before any other content.
Returns True if user is authenticated, False if not.
"""

import streamlit as st
from mdu_engine.auth import generate_otp, store_otp, verify_otp
from mdu_engine.mailer import send_otp_email


def show_login_gate() -> bool:
    """
    Shows login UI if user is not authenticated.
    Returns True if authenticated (app should proceed).
    Returns False if not authenticated (app should stop).

    Usage in app.py — add these 4 lines right after st.set_page_config:

        from login_page import show_login_gate
        if not show_login_gate():
            st.stop()
        # rest of app continues only for authenticated users
    """

    # Already authenticated in this session
    if st.session_state.get("authenticated"):
        return True

    # ── Login page UI ──────────────────────────────────────
    _render_login_header()

    # Two-step flow managed by session state
    step = st.session_state.get("login_step", "enter_email")

    if step == "enter_email":
        _render_email_step()
    elif step == "enter_otp":
        _render_otp_step()

    return False


def _render_login_header():
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                padding:3rem 0 2rem 0;">
        <div style="font-size:2.4rem;font-weight:800;
                    background:linear-gradient(90deg,#1E90FF 0%,#FF6B2B 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;letter-spacing:-0.5px;margin-bottom:6px;">
            MDU Engine
        </div>
        <div style="color:#FF6B2B;font-size:0.75rem;font-weight:600;
                    letter-spacing:0.12em;margin-bottom:8px;">
            DECISION-SUPPORT PLATFORM
        </div>
        <div style="color:#8BA3C7;font-size:0.9rem;text-align:center;
                    max-width:360px;line-height:1.6;">
            Sign in to access your decision history,
            override tracking, and audit trail.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_email_step():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background:#1E3A5F;border:1px solid rgba(30,144,255,0.25);
                    border-radius:12px;padding:28px 24px;margin-bottom:16px;">
            <div style="color:#4AABFF;font-weight:600;font-size:1rem;
                        margin-bottom:4px;">Sign in</div>
            <div style="color:#8BA3C7;font-size:0.8rem;margin-bottom:20px;">
                No password required. We'll email you a code.
            </div>
        </div>
        """, unsafe_allow_html=True)

        name = st.text_input(
            "Your name",
            placeholder="e.g. Priya Sharma",
            key="login_name"
        )
        email = st.text_input(
            "Email address",
            placeholder="e.g. priya@agency.com",
            key="login_email"
        )

        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)

        if st.button("Send Login Code", use_container_width=True, type="primary"):
            name = name.strip()
            email = email.strip().lower()

            if not name:
                st.error("Please enter your name.")
                return
            if not email or "@" not in email:
                st.error("Please enter a valid email address.")
                return

            # Generate and store OTP
            otp = generate_otp()
            store_otp(email, name, otp)

            # Send email
            result = send_otp_email(email, name, otp)

            if result["success"]:
                st.session_state.login_step = "enter_otp"
                st.session_state.login_email_pending = email
                st.session_state.login_name_pending = name
                st.rerun()
            else:
                st.error(f"Could not send email: {result['error']}")

        st.markdown("""
        <div style="text-align:center;margin-top:16px;color:#5A7A9F;font-size:0.75rem;">
            Free to use · Your decisions are saved to your account
        </div>
        """, unsafe_allow_html=True)


def _render_otp_step():
    email = st.session_state.get("login_email_pending", "")
    name  = st.session_state.get("login_name_pending", "")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background:#1E3A5F;border:1px solid rgba(30,144,255,0.25);
                    border-radius:12px;padding:28px 24px;margin-bottom:16px;">
            <div style="color:#4AABFF;font-weight:600;font-size:1rem;
                        margin-bottom:4px;">Check your email</div>
            <div style="color:#8BA3C7;font-size:0.8rem;margin-bottom:4px;">
                We sent a 6-digit code to
            </div>
            <div style="color:#E8F0FE;font-size:0.9rem;font-weight:500;
                        margin-bottom:4px;">
                {email}
            </div>
            <div style="color:#5A7A9F;font-size:0.75rem;">
                Code expires in 10 minutes
            </div>
        </div>
        """, unsafe_allow_html=True)

        otp_input = st.text_input(
            "Enter your 6-digit code",
            placeholder="000000",
            max_chars=6,
            key="otp_input"
        )

        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)

        if st.button("Verify & Sign In", use_container_width=True, type="primary"):
            result = verify_otp(email, otp_input.strip())
            if result["success"]:
                # Set authenticated session
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = result["name"]
                # Clean up login state
                st.session_state.pop("login_step", None)
                st.session_state.pop("login_email_pending", None)
                st.session_state.pop("login_name_pending", None)
                st.rerun()
            else:
                st.error(result["error"])

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        if st.button("← Use a different email", use_container_width=False):
            st.session_state.login_step = "enter_email"
            st.rerun()


def show_user_header():
    """
    Shows a small welcome bar with user name and sign out button.
    Call this at the top of app.py after the login gate passes,
    just before the MDU Engine logo header.
    """
    name  = st.session_state.get("user_name", "")
    email = st.session_state.get("user_email", "")

    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(
            f'<div style="color:#8BA3C7;font-size:0.8rem;padding:4px 0;">'
            f'Signed in as <span style="color:#4AABFF;font-weight:500;">{name}</span>'
            f' &nbsp;·&nbsp; <span style="color:#5A7A9F;">{email}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button("Sign out", key="signout_btn"):
            for key in ["authenticated", "user_email", "user_name"]:
                st.session_state.pop(key, None)
            st.rerun()