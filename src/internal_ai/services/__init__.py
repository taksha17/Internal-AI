"""One function per action, shared by the CLI, the JSON API and the console.

Each function takes the platform and an `Actor`, checks the actor's
permission, does the work through the existing services, and records the
outcome in the audit log. Keeping one code path per action is what makes the
console safe: a rule enforced here holds for every client.
"""
