"""The browser console: every platform step, behind sign-in, roles and gates.

Pages are rendered on the server and htmx only polls job status, so there is
no client-side state to tamper with and no inline script for the content
security policy to allow.
"""
