# Forensic DGP download page

## Product
A simple public landing page that lets Windows users download the verified local
Forensic DGP installer. Visitor mode: Persuade, with a direct download task.

## Scope
P0: product name, primary Windows download, accurate preview/version/size,
requirements, three setup steps, experimental limitation, checksum and licenses.
P1: copy SHA-256 with success/failure feedback.
P2/P3: no account, analytics, forms, backend processing, showcase photos or pricing.

## Journey
Read the product purpose, check Windows/preview requirements, download the EXE,
run setup with network access, open Forensic DGP from the Start Menu.
The page hosts a download; it never accepts face images or video.

## Stack
React + TypeScript + Vite. Cloudflare Workers Static Assets contains both the
built website and the exact verified 6.2 MB executable. No R2 is needed.
Browser downloads remain native; the page must never claim the file finished
downloading merely because its link was clicked.

## Evidence boundary
The installer is an unsigned research preview. Clean-machine and native-camera
release gates remain open. Generated facial details are estimates and do not
verify identity. Preserve non-commercial upstream licensing notices.
