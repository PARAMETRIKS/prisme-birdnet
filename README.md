---
title: Prisme BirdNET
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Prisme BirdNET service

Non-commercial BirdNET audio-species recognition endpoint for Prisme.

- `GET /health` — service health
- `POST /analyze` — multipart upload using `audio` or `file`, plus optional JSON `meta`

BirdNET source code is MIT licensed. BirdNET models are CC BY-NC-SA 4.0 and
must not be used commercially without separate permission.
