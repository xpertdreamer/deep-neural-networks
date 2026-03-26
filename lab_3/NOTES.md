# Notes

*   **Migration from conda to nix:** 
    In this laboratory work, I switched from conda to nix. It felt somewhat redundant to use one package manager to install another, but this was necessary for the environment setup.

*   **Switching to VS Code:**
    I temporarily moved from Spyder to VS Code due to issues with `qtwebengine`. 
    The version required by Spyder is marked as deprecated or insecure in the Nix store, preventing a native build.
    While I tried using the Flatpak version of Spyder, it couldn't access the environment variables/packages due to sandboxing.
    VS Code was chosen as a more stable alternative for this setup.

*   **Task Implementation:**
    Everything was implemented according to my understanding of the requirements (maybe misunderstanding).
