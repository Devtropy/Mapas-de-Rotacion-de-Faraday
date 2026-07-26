import os
import sys

_RAIZ_REPO = os.path.dirname(os.path.abspath(__file__))
if _RAIZ_REPO not in sys.path:
    sys.path.insert(0, _RAIZ_REPO)
