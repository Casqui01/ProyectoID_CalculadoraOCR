import cv2
import pytesseract
import re
import ast
import operator
import numpy as np
import os

# --- Configuración de Tesseract (En Windows) ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- Evaluación segura ---
def evaluar_expresion(expr):
    operadores = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv,
                  ast.USub: operator.neg}
    def _eval(node):
        if isinstance(node, ast.BinOp):
            return operadores[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operadores[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError(f"Operador no permitido: {node}")
    tree = ast.parse(expr, mode='eval')
    return _eval(tree.body)


# --- Preprocesamiento ---
def preprocesar_imagen(img, brillo=0, contraste=1.0):
    img_ajustada = cv2.convertScaleAbs(img, alpha=contraste, beta=brillo)
    gray = cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    return thresh, img_ajustada


# --- Limpiar texto OCR ---
def limpiar_texto(texto):
    reemplazos = {'O': '0', 'o': '0', 'l': '1', 'I': '1', 'x': '*', 'X': '*'}
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)
    return texto.strip().replace(" ", "")


# --- Procesar imagen (OCR + rectángulos) ---
def procesar_imagen(img):
    brillo = cv2.getTrackbarPos("Brillo", "Resultado") - 100
    contraste = cv2.getTrackbarPos("Contraste", "Resultado") / 50.0
    thresh, ajustada = preprocesar_imagen(img, brillo, contraste)

    config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/xX()'
    datos = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
    img_resultado = ajustada.copy()

    for i in range(len(datos['text'])):
        texto = limpiar_texto(datos['text'][i])
        if texto and re.fullmatch(r'[\d+\-*/()]+', texto):
            x, y, w, h = datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i]
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
            try:
                resultado = evaluar_expresion(texto)
                texto_mostrar = f"{texto} = {resultado}"
                cv2.putText(img_resultado, texto_mostrar, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except:
                pass

    cv2.imshow("Resultado", img_resultado)
    cv2.imshow("Preprocesada", thresh)


# --- Procesar cámara en vivo ---
def procesar_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara")

    brillo_inicial = 100
    contraste_inicial = 30

    cv2.namedWindow("Camara")
    cv2.createTrackbar("Brillo", "Camara", brillo_inicial, 200, lambda x: None)
    cv2.createTrackbar("Contraste", "Camara", contraste_inicial, 200, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        brillo = cv2.getTrackbarPos("Brillo", "Camara") - 100
        contraste = cv2.getTrackbarPos("Contraste", "Camara") / 50.0
        thresh, ajustada = preprocesar_imagen(frame, brillo, contraste)

        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/xX()'
        datos = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
        img_resultado = ajustada.copy()

        for i in range(len(datos['text'])):
            texto = limpiar_texto(datos['text'][i])
            if texto and re.fullmatch(r'[\d+\-*/()]+', texto):
                x, y, w, h = datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i]
                cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
                try:
                    resultado = evaluar_expresion(texto)
                    texto_mostrar = f"{texto} = {resultado}"
                    cv2.putText(img_resultado, texto_mostrar, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except:
                    pass

        cv2.imshow("Camara", img_resultado)
        cv2.imshow("Preprocesada", thresh)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Cámara con captura manual ---
def camara_con_captura():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    cv2.namedWindow("Captura OCR")
    print("Presiona ENTER para capturar una foto o ESC para salir...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Captura OCR", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            print("Captura realizada. Analizando imagen...")
            cap.release()
            cv2.destroyWindow("Captura OCR")

            brillo_inicial = 100
            contraste_inicial = 30

            cv2.namedWindow("Resultado")
            cv2.createTrackbar("Brillo", "Resultado", brillo_inicial, 200, lambda x: None)
            cv2.createTrackbar("Contraste", "Resultado", contraste_inicial, 200, lambda x: None)

            while True:
                procesar_imagen(frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cv2.destroyAllWindows()
            return

        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Nueva función: cargar y procesar imagen desde archivo ---
def cargar_y_procesar_imagen():
    ruta = input("Ingrese la ruta de la imagen: ").strip()
    if not os.path.isfile(ruta):
        print("Archivo no encontrado")
        return
    img = cv2.imread(ruta)

    brillo_inicial = 100
    contraste_inicial = 30

    cv2.namedWindow("Resultado")
    cv2.createTrackbar("Brillo", "Resultado", brillo_inicial, 200, lambda x: None)
    cv2.createTrackbar("Contraste", "Resultado", contraste_inicial, 200, lambda x: None)

    while True:
        procesar_imagen(img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


# --- Menú gráfico ---
def menu_grafico():
    menu_img = np.zeros((700, 1200, 3), dtype=np.uint8)
    menu_img[:] = (30, 30, 30)

    cv2.putText(menu_img, "=== CALCULADORA OCR ===", (200, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
    cv2.putText(menu_img, "1 - Cargar imagen desde archivo", (140, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(menu_img, "2 - Usar camara en directo (OCR vivo)", (140, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(menu_img, "3 - Camara (capturar con ENTER para analizar)", (140, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(menu_img, "0 - Salir del programa", (140, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(menu_img, "Presiona un numero para elegir una opcion", (120, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 255, 180), 2)

    cv2.namedWindow("Menu", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Menu", 1200, 700)
    cv2.imshow("Menu", menu_img)

    while True:
        key = cv2.waitKey(0)
        if key == ord('1'):
            cv2.destroyWindow("Menu")
            cargar_y_procesar_imagen()
            break
        elif key == ord('2'):
            cv2.destroyWindow("Menu")
            procesar_camara()
            break
        elif key == ord('3'):
            cv2.destroyWindow("Menu")
            camara_con_captura()
            break
        elif key == ord('0'):
            cv2.destroyWindow("Menu")
            break


# --- Ejecutar ---
if __name__ == "__main__":
    menu_grafico()
