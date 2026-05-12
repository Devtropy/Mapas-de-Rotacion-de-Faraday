#  Simulación Numérica de Campos Magnéticos y Rotación de Faraday

## Resumen del Proyecto

Este proyecto es una simulación numérica de alto rendimiento desarrollada para modelar la estructura de campos magnéticos turbulentos y sus firmas observacionales en el Medio Intracúmulo (ICM). Utilizando la arquitectura paralela de **NVIDIA CUDA**, este proyecto permite simular la naturaleza estocástica de los campos magnéticos galácticos y generar mapas sintéticos de Medidas de Rotación (RM) e intensidad de emisión sincrotrón.

La simulación genera campos gaussianos isotrópicos bajo la restricción de divergencia nula ($\nabla \cdot \mathbf{B} = 0$) y procesa poblaciones de electrones relativistas para calcular los parámetros de Stokes $Q$ y $U$ finales.

## Características de la Simulación

* **Generación de Campos Turbulentos:** Implementación en el espacio de Fourier basada en un espectro de potencia $P(k) \propto k^{-n}$.
* **Perfil de Densidad Térmica:** El campo se escala radialmente según un modelo $\beta$ para la densidad electrónica $n_e(r)$.
* **Cómputo Acelerado (CUDA):** Kernels optimizados para la integración por línea de visión en mallas tridimensionales de alta resolución.
* **Análisis de Polarización:** Integración de efectos de depolarización y rotación de Faraday para la obtención de mapas de polarización sintética.

## Fundamentos Físicos

El modelo se basa en los siguientes pilares:

1. **Perfil de Densidad Electrónica:**
   $$n_{e}(r)=n_{0}\left(1+\frac{r^{2}}{r_{e}^{2}}\right)^{-3\beta/2}$$
   
2. **Medida de Rotación (RM):**
   $$RM(x,y) = \int_{0}^{L} 812 \cdot n_e(r) B_{||}(l) \, dl$$
   
3. **Emisividad Sincrotrón:**
   $$j_{\nu} \propto n_{rel}(r) \cdot B_{\perp}^{(p+1)/2} \cdot \nu^{-(p-1)/2}$$

## Autores

* **Ivan Acosta** .
* **Diego Rivera**.
* **Angel Rivera**.

