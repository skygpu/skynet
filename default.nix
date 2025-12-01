with (import <nixpkgs> {});
let
  glibStorePath = lib.getLib glib;
  zstdStorePath = lib.getLib zstd;
  libGLStorePath = lib.getLib libGL;
in
stdenv.mkDerivation {
  name = "skynet";
  buildInputs = [
    # System requirements.
    glib
    zstd
    libGL
    libgcc.lib
    # Python requirements.
    python313Full
    python313Packages.uv
  ];
  src = null;
  shellHook = ''
    set -e

    LIB_GCC_PATH="${libgcc.lib}/lib"
    GLIB_PATH="${glibStorePath}/lib"
    ZSTD_PATH="${zstdStorePath}/lib"
    LIBGL_PATH="${libGLStorePath}/lib"


    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIB_GCC_PATH"
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GLIB_PATH"
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ZSTD_PATH"
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBGL_PATH"

    export LD_LIBRARY_PATH
  '';
}
