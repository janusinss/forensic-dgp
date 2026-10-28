# Restoration and reviewed completion on main

The `integrated_app/` folder contains the application published on `IMPLEMENTATION` at commit `6059fc32acbe04800241fc5be15f07549d30b0b9`. Its Conditioned SFT V2 (`p2_u0224`) restoration and optional reviewed CodeFormer R1 completion use separate model assets and workers. The full research application and its existing launchers remain in place.

On Windows, from the repository root, run `install-integrated.bat` once and then `start-integrated.bat`. Setup needs network access and 64-bit Python 3.12 or 3.13; all inference after setup is local. If the port is occupied, run `start-integrated.bat --port 0`. For details see `integrated_app/README.MD` and `integrated_app/COMPLETION_INSTALL.md`.

Review the proposed covering area, correct it as needed and confirm it before generating. In the retained nine-case development check the automatic detector proposed empty masks in all cases, including seven covered cases. Manual mask correction is necessary for those misses; the detector is not qualified. Completion is a plausible estimate, not proof of hidden identity. Independent final review remains outstanding.

To update this checkout, preserve any local changes and use `git pull --ff-only origin main` from the repository root; then rerun `install-integrated.bat` if dependencies or assets have changed. Never use the IMPLEMENTATION checkout update command on main.
