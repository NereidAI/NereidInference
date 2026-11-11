<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Security Policy

## Supported Versions

This project is in the 0.x development phase. Security fixes are applied to
the rolling `0.0.0-dev` channel and the latest tagged release; older tags do
not receive backports.

| Channel | Supported |
| --- | --- |
| Latest git tag (`v*`) and the `0.0.0-dev` image channel | Yes |
| Earlier tags | No |

## Reporting a Vulnerability

If you discover a security vulnerability in NereidAI software, **please do
not open a public GitHub issue**. Report it privately so it can be triaged
and fixed before public disclosure.

- **Email**: jiekechoo@gmail.com
- **Encryption**: PGP is preferred; request the public key by email.
- **Acknowledgement target**: within 72 hours of receipt.

Please include:

- a description of the vulnerability and its impact;
- reproduction steps (a proof of concept is helpful);
- the affected version/tag or image digest and the deployment context
  (direct internet, jump-host, or fully offline).

## Disclosure workflow

1. Triage and confirmation, followed by a private fix on a security branch.
2. A new signed and protected `v*` tag publishing the fixed image with the
   standard three-tag scheme; release notes describe the issue without
   exploitation details.
3. Public disclosure after a fix is available, or after 90 days if no fix
   can be shipped, whichever comes first.

## Hardening expectations

- Deployment secrets are injected at install time via Kubernetes Secrets and
  environment variables; tracked configuration files carry empty
  placeholders only and must never contain real credentials.
- Production deployments serve traffic over HTTPS (TLS 1.2 or later).
- Report any committed credential as a vulnerability regardless of age;
  repository history cannot be un-published once a repository is public.
