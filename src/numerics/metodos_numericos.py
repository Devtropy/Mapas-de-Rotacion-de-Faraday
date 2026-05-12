def simpson13(funcion, a, b, n):
    h = (b - a) / (2 * n)
    integral = funcion(a) + funcion(b)
    suma_impar = 0.0
    suma_par = 0.0
    for i in range(1, n + 1):
        suma_impar += funcion(a + (2 * i - 1) * h)
    for i in range(1, n):
        suma_par += funcion(a + 2 * i * h)
    integral = (h / 3) * (integral + 4 * suma_impar + 2 * suma_par)
    return integral
