from dataclasses import dataclass


@dataclass
class Parametros:
    n0: float = 10e-3
    rc: float = 400
    beta: float = 0.6
    B0: float = 1e-6
    mu: float = 0.4
    p: float = 3
    nu: float = 1.4
    n: float = 11 / 3
    lambda_min: float = 6
    lambda_max: float = 768
    dx: float = 3
    N: int = 512
