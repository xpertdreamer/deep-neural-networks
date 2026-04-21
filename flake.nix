{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1"; # unstable Nixpkgs

  outputs =
    { self, ... }@inputs:

    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs { inherit system; };
          }
        );

      /*
        Change this value ({major}.{min}) to
        update the Python virtual-environment
        version. When you do this, make sure
        to delete the `.venv` directory to
        have the hook rebuild it for the new
        version, since it won't overwrite an
        existing one. After this, reload the
        development shell to rebuild it.
        You'll see a warning asking you to
        do this when version mismatches are
        present. For safety, removal should
        be a manual step, even if trivial.
      */
      version = "3.13";
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        let
          concatMajorMinor =
            v:
            pkgs.lib.pipe v [
              pkgs.lib.versions.splitVersion
              (pkgs.lib.sublist 0 2)
              pkgs.lib.concatStrings
            ];

          python = pkgs."python${concatMajorMinor version}";
        in
        {
          default = pkgs.mkShellNoCC {
            venvDir = ".venv";

            postShellHook = ''
              venvVersionWarn() {
              	local venvVersion
              	venvVersion="$("$venvDir/bin/python" -c 'import platform; print(platform.python_version())')"

              	[[ "$venvVersion" == "${python.version}" ]] && return

              	cat <<EOF
              Warning: Python version mismatch: [$venvVersion (venv)] != [${python.version}]
                       Delete '$venvDir' and reload to rebuild for version ${python.version}
              EOF
              }

              venvVersionWarn
            '';

            packages = with python.pkgs; [
              venvShellHook
              pip

              # === Научные вычисления ===
              numpy
              scipy
              matplotlib
              sympy
              pandas
              numpydoc

              # === Машинное обучение ===
              scikit-learn
              torch
              torchvision
              joblib

              # === Jupyter ===
              jupyter
              jupyter-client
              jupyter-core
              ipykernel
              ipython
              nbconvert
              nbformat

              # === Асинхронность ===
              aiohttp
              aiodns
              asyncssh
              uvloop

              # === Базы данных ===
              mysqlclient
              psycopg2

              # === Инструменты разработки ===
              black
              isort
              flake8
              pylint
              mypy
              ruff

              # === LSP ===
              python-lsp-server
              python-lsp-black
              python-lsp-ruff

              # === Документация ===
              sphinx
              docutils

              # === Веб ===
              requests
              click
              jinja2
              pyyaml

              # === Утилиты ===
              tqdm
              rich
              packaging
              psutil
              watchdog

              # === Обработка данных ===
              pillow
              beautifulsoup4

              # === Дополнительно ===
              pytest
              pytest-cov
              tox
            ];
          };
        }
      );
    };
}
