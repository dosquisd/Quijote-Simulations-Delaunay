{
  description = "Dev shell para Quijote-Simulations-Delaunay (Python 3.12 + uv + Jupyter + Pylians con clang)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };

      python = pkgs.python312;

      clang = pkgs.clang_20;
      llvm = pkgs.llvm_20;

      gsl = pkgs.gsl;
      fftw = pkgs.fftw;
      fftwFloat = pkgs.fftwFloat;
      hdf5 = pkgs.hdf5.dev;
      zlib = pkgs.zlib;
      pkg-config = pkgs.pkg-config;
      libgcc = pkgs.libgcc;
      gcc-unwrapped-lib = pkgs.gcc-unwrapped.lib;
      openmp = pkgs.llvmPackages_20.openmp;

      includeFlags = pkgs.lib.concatStringsSep " " [
        "-I${gsl}/include"
        "-I${fftw.dev}/include"
        "-I${fftwFloat.dev}/include"
        "-I${hdf5}/include"
        "-I${zlib.dev}/include"
        "-I${openmp.dev}/include"
      ];

      libFlags = pkgs.lib.concatStringsSep " " [
        "-L${gsl}/lib"
        "-L${fftw}/lib"
        "-L${fftwFloat}/lib"
        "-L${hdf5}/lib"
        "-L${zlib}/lib"
        "-L${openmp}/lib"
      ];

      ldLibraryPath = pkgs.lib.concatStringsSep ":" [
        "${gsl}/lib"
        "${fftw}/lib"
        "${fftwFloat}/lib"
        "${hdf5}/lib"
        "${zlib}/lib"
        "${llvm.lib}/lib"
        "${libgcc}/lib"
        "${gcc-unwrapped-lib}/lib"
        "${openmp}/lib"
      ];

      pkgConfigPath = pkgs.lib.concatStringsSep ":" [
        "${gsl}/lib/pkgconfig"
        "${fftw.dev}/lib/pkgconfig"
        "${fftwFloat.dev}/lib/pkgconfig"
        "${hdf5}/lib/pkgconfig"
      ];

      rpathLibs = pkgs.lib.concatStringsSep ":" [
        "${libgcc}/lib"
        "${gcc-unwrapped-lib}/lib"
        "${pkgs.glibc}/lib"
        "${zlib}/lib"
        "${gsl}/lib"
        "${fftw}/lib"
        "${fftwFloat}/lib"
        "${pkgs.hdf5}/lib"
        "${openmp}/lib"
        "${llvm.lib}/lib"
      ];
    in
    {
      devShells.default = pkgs.mkShell {
        nativeBuildInputs = [ pkg-config ];

        buildInputs = [
          python
          pkgs.uv
          clang
          llvm
          gsl
          fftw
          fftwFloat
          pkgs.hdf5
          zlib
          libgcc
          gcc-unwrapped-lib
          openmp
          pkgs.patchelf
        ];

        env = {
          CC = "${clang}/bin/clang";
          CXX = "${clang}/bin/clang++";
          CPP = "${clang}/bin/clang-cpp";
          LD = "${clang}/bin/ld";
          AR = "${llvm}/bin/llvm-ar";
          NM = "${llvm}/bin/llvm-nm";
          RANLIB = "${llvm}/bin/llvm-ranlib";

          CFLAGS = includeFlags;
          CXXFLAGS = includeFlags;
          CPPFLAGS = includeFlags;
          LDFLAGS = libFlags;

          HDF5_DIR = "${hdf5}";
          FFTW_DIR = "${fftw}";
          GSL_DIR = "${gsl}";

          PKG_CONFIG_PATH = pkgConfigPath;

          NIX_LD_LIBRARY_PATH = ldLibraryPath;
          LD_LIBRARY_PATH = ldLibraryPath;

          UV_PYTHON = "${python}/bin/python3.12";
          UV_PYTHON_PREFERENCE = "only-system";
        };

        shellHook = ''
          # mkShell (stdenv) fija CC/CXX a gcc; forzamos clang 20
          export CC="${clang}/bin/clang"
          export CXX="${clang}/bin/clang++"
          export CPP="${clang}/bin/clang-cpp"
          export LD="${clang}/bin/ld"
          export AR="${llvm}/bin/llvm-ar"
          export NM="${llvm}/bin/llvm-nm"
          export RANLIB="${llvm}/bin/llvm-ranlib"

          echo ""
          echo "  Quijote-Simulations-Delaunay — dev shell"
          echo "  Python: $(${python}/bin/python3.12 --version)"
          echo "  CC: $CC"
          echo "  uv:   $(uv --version)"
          echo ""

          if [ ! -d ".venv" ]; then
            echo "[shellHook] Creando .venv con uv..."
            uv venv --python ${python}/bin/python3.12 .venv
          fi

          # Las build constraints de Cython/numpy para Pylians se leen de
          # pyproject.toml ([tool.uv] build-constraint-dependencies)

          echo "[shellHook] Sincronizando dependencias con uv..."
          uv sync

          # NixOS: los wheels manylinux compilados (.so) buscan libstdc++/libgcc en
          # rutas estandar que no existen en NixOS. Les metemos rpath a las librerias
          # de Nix para que el .venv funcione TAMBIEN fuera de `nix develop`
          # (VS Code, jupyter notebook directo, etc.).
          if command -v patchelf >/dev/null 2>&1; then
            echo "[shellHook] Parcheando wheels del venv con rpath (para uso fuera de nix develop)..."
            for f in $(find .venv/lib -name "*.so*"); do
              orig="$(patchelf --print-rpath "$f" 2>/dev/null || true)"
              case ":$orig:" in
                *":${rpathLibs}:"*) continue ;;
              esac
              if [ -z "$orig" ]; then
                patchelf --set-rpath "${rpathLibs}" "$f" 2>/dev/null || true
              else
                # mantener el rpath original del wheel (p.ej. $ORIGIN/../xx.libs) y anadir las libs de Nix
                patchelf --set-rpath "$orig:${rpathLibs}" "$f" 2>/dev/null || true
              fi
            done
            # las libs bundleadas (<pkg>.libs) se dependen transitivamente (p.ej. libfftw3l_omp -> libgomp);
            # glibc no propaga el RUNPATH, asi que les damos rpath $ORIGIN para que se encuentren entre si
            for d in $(find .venv/lib -type d -name "*.libs"); do
              for f in "$d"/*.so*; do
                [ -e "$f" ] || continue
                orig="$(patchelf --print-rpath "$f" 2>/dev/null || true)"
                case "$orig" in
                  *ORIGIN*) continue ;;
                esac
                patchelf --set-rpath "\$ORIGIN:$orig" "$f" 2>/dev/null || true
              done
            done
            echo "[shellHook] Wheels parcheados."
          fi

          source .venv/bin/activate

          echo "[shellHook] Registrando kernel de Jupyter 'delaunay'..."
          python -m ipykernel install --user --name delaunay --display-name "Python (delaunay)" 2>/dev/null || true

          echo "[shellHook] Listo. Ejecuta: jupyter notebook"
          echo ""
        '';
      };
    });
}
