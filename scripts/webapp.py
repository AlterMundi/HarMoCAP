#!/usr/bin/env python
"""Lanza la interfaz web local de HarMoCAP.

Uso:
    python scripts/webapp.py            # abre localhost:7860 en el navegador
    python scripts/webapp.py --port 8000
    python scripts/webapp.py --share    # link temporal público (Gradio)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harmocap.webapp.app import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="crea un link temporal público (por defecto NO: todo local)")
    args = ap.parse_args()
    main(share=args.share, port=args.port)
